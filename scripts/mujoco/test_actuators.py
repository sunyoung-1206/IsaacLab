"""
MuJoCo 액추에이터 단위 테스트 (PD 제어)

모드:
  pd_hold  - action=0 (default pose 유지). PD 제어 안정성 확인.
  pd_step  - 한 관절에 step 목표를 주고 추종 성능 측정.

실행 예시:
  python test_actuators.py --scene ~/mujoco_menagerie/unitree_go2/go2_scene.xml --mode pd_hold
  python test_actuators.py --scene ~/mujoco_menagerie/unitree_go2/go2_scene.xml --mode pd_step --step_joint 4 --step_angle 0.3
"""

import argparse
import time
import numpy as np
import mujoco
import mujoco.viewer

# ── 공통 파라미터 ──────────────────────────────────────────────────────────────
KP           = 25.0
KD           = 0.6
EFFORT_LIMIT = 23.5    # N·m
SIM_DT       = 0.005   # s
DECIMATION   = 4       # policy 50Hz
ACTION_SCALE = 0.25

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
    ctrl_to_policy = np.array([POLICY_JOINTS.index(n) for n in actuator_jnames])
    policy_to_ctrl = np.argsort(ctrl_to_policy)

    print("[관절 매핑]")
    for pi, name in enumerate(POLICY_JOINTS):
        print(f"  policy[{pi:2d}] → mj[{policy_to_mj[pi]:2d}]  {name}")

    return policy_to_mj, policy_to_ctrl


# ── 초기화 ────────────────────────────────────────────────────────────────────

def reset_sim(model, data, policy_to_mj):
    mujoco.mj_resetData(model, data)
    for j in range(12):
        data.qpos[7 + policy_to_mj[j]] = DEFAULT_JOINT_POS[j]

    data.qpos[2] = 1.0
    mujoco.mj_forward(model, data)
    foot_names = {"FL_foot", "FR_foot", "RL_foot", "RR_foot", "FL", "FR", "RL", "RR"}
    foot_zs = [data.geom_xpos[i, 2] for i in range(model.ngeom)
                if model.geom(i).name in foot_names]
    data.qpos[2] = (1.0 - min(foot_zs) + 0.002) if foot_zs else 0.35
    mujoco.mj_forward(model, data)
    print(f"[초기화] base z = {data.qpos[2]:.3f} m")


# ── PD 토크 ───────────────────────────────────────────────────────────────────

def pd_torque(q, qdot, q_target):
    tau = KP * (q_target - q) + KD * (0.0 - qdot)
    return np.clip(tau, -EFFORT_LIMIT, EFFORT_LIMIT)


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene",      required=True)
    parser.add_argument("--mode",       default="pd_hold", choices=["pd_hold", "pd_step"])
    parser.add_argument("--duration",   type=float, default=10.0)
    parser.add_argument("--step_joint", type=int,   default=4,
                        help="pd_step 모드: step 입력 줄 관절 인덱스 (policy 순서, 기본=4=FL_thigh)")
    parser.add_argument("--step_angle", type=float, default=0.3,
                        help="pd_step 모드: default_pos에서의 추가 각도 [rad]")
    parser.add_argument("--step_time",  type=float, default=2.0,
                        help="pd_step 모드: step 입력 시각 [s]")
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

    print(f"\n{'t(s)':>6}  {'h(m)':>6}  {'q_err_max':>9}  {'tau_max':>8}  {'qdot_max':>9}")
    print("-" * 55)

    log_every = 50
    step = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time < args.duration:
            t_start = time.perf_counter()

            q_policy    = data.qpos[7:][policy_to_mj]
            qdot_policy = data.qvel[6:][policy_to_mj]

            q_target = DEFAULT_JOINT_POS.copy()
            if args.mode == "pd_step" and data.time >= args.step_time:
                q_target[args.step_joint] += args.step_angle

            for _ in range(DECIMATION):
                q_now    = data.qpos[7:][policy_to_mj]
                qdot_now = data.qvel[6:][policy_to_mj]
                tau      = pd_torque(q_now, qdot_now, q_target)
                data.ctrl[policy_to_ctrl] = tau
                mujoco.mj_step(model, data)

            step += 1
            viewer.sync()

            if step % log_every == 0:
                q_policy    = data.qpos[7:][policy_to_mj]
                qdot_policy = data.qvel[6:][policy_to_mj]
                h         = data.qpos[2]
                q_err_max = np.abs(q_policy - DEFAULT_JOINT_POS).max()
                tau_max   = np.abs(tau).max()
                qdot_max  = np.abs(qdot_policy).max()
                print(f"{data.time:6.2f}  {h:6.3f}  {q_err_max:9.4f}  {tau_max:8.3f}  {qdot_max:9.4f}")

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
