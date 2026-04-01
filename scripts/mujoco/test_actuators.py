"""
MuJoCo 액추에이터 단위 테스트

모드:
  pd_hold       - 순수 PD 토크, action=0 (default pose 유지). PD 제어가 안정적인지 확인.
  electric_hold - 전기모터 ODE (dI/dt), action=0. 전류 기반 토크가 안정적인지 확인.
  pd_step       - 한 관절에 step 목표를 주고 추종 성능 측정.

실행 예시:
  python test_actuators.py --scene ~/mujoco_menagerie/unitree_go2/go2_scene.xml --mode pd_hold
  python test_actuators.py --scene ~/mujoco_menagerie/unitree_go2/go2_scene.xml --mode electric_hold
  python test_actuators.py --scene ~/mujoco_menagerie/unitree_go2/go2_scene.xml --mode pd_step --step_joint 4 --step_angle 0.3
"""

import argparse
import time
import numpy as np
import mujoco
import mujoco.viewer
from scipy.integrate import solve_ivp

# ── 공통 파라미터 ──────────────────────────────────────────────────────────────
KP           = 25.0
KD           = 0.6
EFFORT_LIMIT = 23.5    # N·m
SIM_DT       = 0.005   # s
DECIMATION   = 4       # policy 50Hz
ACTION_SCALE = 0.25

# 전기모터 파라미터 (electric_motor.py 기준)
Kt          = 0.128    # 토크 상수  [N·m/A]
Ke          = 0.128    # 역기전력  [V·s/rad]
R_motor     = 0.3      # 권선 저항  [Ω]
L_motor     = 1e-4     # 인덕턴스  [H]
V_MAX       = 24.0     # 공급 전압  [V]
GEAR_RATIO  = 6.33

# Policy 관절 순서 (PhysX/Isaac Lab 기준)
POLICY_JOINTS = [
    "FL_hip_joint",   "FR_hip_joint",   "RL_hip_joint",   "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint",  "FR_calf_joint",  "RL_calf_joint",  "RR_calf_joint",
]

DEFAULT_JOINT_POS = np.array([
     0.1, -0.1,  0.1, -0.1,
     0.8,  0.8,  1.0,  1.0,
    -1.5, -1.5, -1.5, -1.5,
], dtype=np.float64)


# ── 매핑 ──────────────────────────────────────────────────────────────────────

def build_mappings(model):
    mj_joint_names = [
        model.joint(i).name
        for i in range(model.njnt)
        if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
    ]
    policy_to_mj = np.array([mj_joint_names.index(n) for n in POLICY_JOINTS])

    actuator_jnames = [
        model.joint(model.actuator_trnid[i, 0]).name
        for i in range(model.nu)
    ]
    ctrl_to_policy  = np.array([POLICY_JOINTS.index(n) for n in actuator_jnames])
    policy_to_ctrl  = np.argsort(ctrl_to_policy)

    print("[관절 매핑]")
    for pi, name in enumerate(POLICY_JOINTS):
        print(f"  policy[{pi:2d}] → mj[{policy_to_mj[pi]:2d}]  {name}")

    return policy_to_mj, policy_to_ctrl


# ── 초기화 ────────────────────────────────────────────────────────────────────

def reset_sim(model, data, policy_to_mj):
    mujoco.mj_resetData(model, data)
    for j in range(12):
        data.qpos[7 + policy_to_mj[j]] = DEFAULT_JOINT_POS[j]

    # FK로 발 최저점 계산 → base 높이 자동 설정
    data.qpos[2] = 1.0
    mujoco.mj_forward(model, data)
    foot_names = {"FL_foot", "FR_foot", "RL_foot", "RR_foot", "FL", "FR", "RL", "RR"}
    foot_zs = [data.geom_xpos[i, 2] for i in range(model.ngeom)
                if model.geom(i).name in foot_names]
    if foot_zs:
        data.qpos[2] = 1.0 - min(foot_zs) + 0.002
    else:
        data.qpos[2] = 0.35
    mujoco.mj_forward(model, data)
    print(f"[초기화] base z = {data.qpos[2]:.3f} m")


# ── PD 토크 ───────────────────────────────────────────────────────────────────

def pd_torque(q, qdot, q_target):
    tau = KP * (q_target - q) + KD * (0.0 - qdot)
    return np.clip(tau, -EFFORT_LIMIT, EFFORT_LIMIT)


# ── 전기모터: V_cmd 계산 ───────────────────────────────────────────────────────

def compute_V_cmd(q, qdot, q_target):
    """PD 역산으로 V_cmd 계산 (electric_motor.py와 동일)."""
    tau_des = KP * (q_target - q) + KD * (0.0 - qdot)
    tau_des = np.clip(tau_des, -EFFORT_LIMIT, EFFORT_LIMIT)
    I_des   = tau_des / (Kt * GEAR_RATIO)
    omega   = qdot * GEAR_RATIO
    V_cmd   = I_des * R_motor + Ke * omega
    return np.clip(V_cmd, -V_MAX, V_MAX)


# ── 전기모터: 전류 ODE 적분 ────────────────────────────────────────────────────

def integrate_current(I_prev, V_cmd, omega, dt):
    """
    dI/dt = (V_cmd - R*I - Ke*omega) / L  를 RK4로 적분.
    tau_e = L/R ≈ 0.33ms → dt=5ms 동안 여러 내부 스텝 필요.
    """
    n_sub = max(5, int(dt / (L_motor / R_motor * 0.3)))  # sub-steps
    dt_sub = dt / n_sub

    I = I_prev.copy()
    for _ in range(n_sub):
        dI = (V_cmd - R_motor * I - Ke * omega) / L_motor
        I  = I + dI * dt_sub  # Euler (sub-step 충분히 작음)

    I_max = V_MAX / R_motor
    return np.clip(I, -I_max, I_max)


# ── 로깅 헤더 ──────────────────────────────────────────────────────────────────

def log_header(mode):
    if mode == "electric_hold":
        print(f"\n{'t(s)':>6}  {'h(m)':>6}  {'q_err_max':>9}  {'tau_max':>8}  {'I_mean':>7}  {'I_max':>7}  {'V_mean':>7}")
        print("-" * 65)
    else:
        print(f"\n{'t(s)':>6}  {'h(m)':>6}  {'q_err_max':>9}  {'tau_max':>8}  {'qdot_max':>9}")
        print("-" * 55)


def log_row(data, mode, q_policy, qdot_policy, tau, I=None, V_cmd=None):
    h = data.qpos[2]
    q_err_max = np.abs(q_policy - DEFAULT_JOINT_POS).max()
    tau_max   = np.abs(tau).max()
    qdot_max  = np.abs(qdot_policy).max()
    t = data.time
    if mode == "electric_hold" and I is not None:
        print(f"{t:6.2f}  {h:6.3f}  {q_err_max:9.4f}  {tau_max:8.3f}  "
              f"{np.abs(I).mean():7.3f}  {np.abs(I).max():7.3f}  {np.abs(V_cmd).mean():7.3f}")
    else:
        print(f"{t:6.2f}  {h:6.3f}  {q_err_max:9.4f}  {tau_max:8.3f}  {qdot_max:9.4f}")


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene",      required=True)
    parser.add_argument("--mode",       default="pd_hold",
                        choices=["pd_hold", "electric_hold", "pd_step"])
    parser.add_argument("--duration",   type=float, default=10.0)
    parser.add_argument("--step_joint", type=int,   default=4,
                        help="pd_step 모드: 스텝 입력 줄 관절 인덱스 (policy 순서, 기본=4=FL_thigh)")
    parser.add_argument("--step_angle", type=float, default=0.3,
                        help="pd_step 모드: default_pos에서의 추가 각도 [rad]")
    parser.add_argument("--step_time",  type=float, default=2.0,
                        help="pd_step 모드: 스텝 입력 시각 [s]")
    args = parser.parse_args()

    print(f"[MODE] {args.mode}")
    model = mujoco.MjModel.from_xml_path(args.scene)
    data  = mujoco.MjData(model)
    if abs(model.opt.timestep - SIM_DT) > 1e-6:
        print(f"[경고] timestep {model.opt.timestep} → {SIM_DT}")
        model.opt.timestep = SIM_DT

    # Isaac Lab 매칭: damping / armature / frictionloss 제거
    for i in range(model.njnt):
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        dof = model.jnt_dofadr[i]
        model.dof_damping[dof]      = 0.0
        model.dof_armature[dof]     = 0.0
        model.dof_frictionloss[dof] = 0.0
    print("[INFO] joint damping / armature / frictionloss → 0")

    policy_to_mj, policy_to_ctrl = build_mappings(model)
    reset_sim(model, data, policy_to_mj)

    # 전기모터 전류 상태 초기화
    I_state = np.zeros(12, dtype=np.float64)

    log_header(args.mode)
    log_every = 50  # policy step 단위 (1초)

    step = 0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time < args.duration:
            t_start = time.perf_counter()

            # ── 현재 관절 상태 (policy 순서) ──────────────────────────────────
            q_policy    = data.qpos[7:][policy_to_mj]
            qdot_policy = data.qvel[6:][policy_to_mj]

            # ── 목표 관절 위치 결정 ───────────────────────────────────────────
            q_target = DEFAULT_JOINT_POS.copy()
            if args.mode == "pd_step" and data.time >= args.step_time:
                q_target[args.step_joint] += args.step_angle

            # ── decimation 루프 ───────────────────────────────────────────────
            for _ in range(DECIMATION):
                q_now    = data.qpos[7:][policy_to_mj]
                qdot_now = data.qvel[6:][policy_to_mj]

                if args.mode in ("pd_hold", "pd_step"):
                    # ── PD 토크 제어 ─────────────────────────────────────────
                    tau = pd_torque(q_now, qdot_now, q_target)
                    V_cmd_log = np.zeros(12)

                else:  # electric_hold
                    # ── 전기모터 ODE ─────────────────────────────────────────
                    V_cmd     = compute_V_cmd(q_now, qdot_now, q_target)
                    omega_now = qdot_now * GEAR_RATIO
                    I_state   = integrate_current(I_state, V_cmd, omega_now, SIM_DT)
                    tau       = np.clip(Kt * I_state * GEAR_RATIO, -EFFORT_LIMIT, EFFORT_LIMIT)
                    V_cmd_log = V_cmd

                data.ctrl[policy_to_ctrl] = tau
                mujoco.mj_step(model, data)

            step += 1
            viewer.sync()

            # ── 로깅 ─────────────────────────────────────────────────────────
            if step % log_every == 0:
                q_policy    = data.qpos[7:][policy_to_mj]
                qdot_policy = data.qvel[6:][policy_to_mj]
                log_row(data, args.mode, q_policy, qdot_policy, tau,
                        I=I_state, V_cmd=V_cmd_log)

            # ── 낙하 감지 ────────────────────────────────────────────────────
            if data.qpos[2] < 0.05:
                print(f"[넘어짐] t={data.time:.2f}s  h={data.qpos[2]:.3f}m")
                break

            elapsed  = time.perf_counter() - t_start
            sleep_dt = SIM_DT * DECIMATION - elapsed
            if sleep_dt > 0:
                time.sleep(sleep_dt)

    print(f"\n[완료] {step} steps  t={data.time:.2f}s  최종 h={data.qpos[2]:.3f}m")


if __name__ == "__main__":
    main()
