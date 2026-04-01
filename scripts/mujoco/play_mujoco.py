"""
MuJoCo play script - Unitree Go2 Isaac Lab 정책 재현

액추에이터 모드:
  --actuator pd       (기본) 순수 PD 토크 제어 (Isaac Lab DCMotor 재현)
  --actuator electric 전기모터 ODE 통합
                      dI/dt = (V_cmd - R·I - Ke·ω) / L
                      τ = Kt · I · gr
                      ω는 매 스텝 MuJoCo에서 읽어옴 → omega overwrite 없음

고장 시뮬레이션 (--actuator electric 전용):
  --fault_joint  관절 인덱스 (policy 순서, 기본=10=RL_calf)
  --fault_R      고장 시 저항 [Ω] (기본=3.0)
  --fault_start  고장 시작 시각 [s] (기본=5.0)
  --fault_dur    고장 전이 시간 [s] (기본=5.0)

실행 예시:
  # PD 모드
  python play_mujoco.py --policy policy.pt --scene go2_scene.xml

  # 전기모터 모드
  python play_mujoco.py --policy policy.pt --scene go2_scene.xml --actuator electric

  # 전기모터 + 고장
  python play_mujoco.py --policy policy.pt --scene go2_scene.xml \\
      --actuator electric --fault_joint 10 --fault_R 3.0 --fault_start 5.0
"""

import argparse
import os
import time
import numpy as np
import torch
import mujoco
import mujoco.viewer

# ─── Isaac Lab 파라미터 ────────────────────────────────────────────────────────
KP                = 25.0
KD                = 0.6
EFFORT_LIMIT      = 23.5   # N·m
SATURATION_EFFORT = 23.5   # N·m  (DCMotorCfg saturation_effort)
VELOCITY_LIMIT    = 30.0   # rad/s (DCMotorCfg velocity_limit)
SIM_DT       = 0.005   # s  (velocity_env_cfg.py sim.dt와 동일)
DECIMATION   = 4       # → policy 50Hz
ACTION_SCALE = 0.25  # env.yaml 확인: 두 정책 모두 scale=0.25로 학습됨

# ─── 전기모터 파라미터 (electric_motor.py 기준) ───────────────────────────────
Kt         = 0.128    # 토크 상수   [N·m/A]
Ke         = 0.128    # 역기전력    [V·s/rad]
R_NOM      = 0.3      # 정상 저항   [Ω]
L_MOTOR    = 1e-4     # 인덕턴스    [H]
V_MAX      = 24.0     # 공급 전압   [V]
GEAR_RATIO = 6.33

# ─── 관절 정의 ────────────────────────────────────────────────────────────────
POLICY_JOINTS = [
    "FL_hip_joint",   "FR_hip_joint",   "RL_hip_joint",   "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint",  "FR_calf_joint",  "RL_calf_joint",  "RR_calf_joint",
]

# 발 geom 이름 (Go2 menagerie 기준: sphere geom이 calf body에 붙어있음)
FOOT_GEOM_NAMES = ["FL", "FR", "RL", "RR"]
FOOT_NAMES      = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]  # 저장·플롯용 레이블

DEFAULT_JOINT_POS = np.array([
     0.1, -0.1,  0.1, -0.1,
     0.8,  0.8,  1.0,  1.0,
    -1.5, -1.5, -1.5, -1.5,
], dtype=np.float64)


# ─── 유틸리티 ─────────────────────────────────────────────────────────────────

def quat_rotate_inverse(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """월드 → 바디 프레임 변환. MuJoCo quat 형식: (w,x,y,z)."""
    q = quat_wxyz / np.linalg.norm(quat_wxyz)
    w, x, y, z = q
    R = np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])
    return R.T @ vec


def build_joint_mapping(model):
    mj_names = [model.joint(i).name for i in range(model.njnt)
                if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
    if len(mj_names) != 12:
        raise ValueError(f"관절 수 오류: {len(mj_names)}개  {mj_names}")

    policy_to_mj = np.array([mj_names.index(n) for n in POLICY_JOINTS])

    act_jnames      = [model.joint(model.actuator_trnid[i,0]).name for i in range(model.nu)]
    ctrl_to_policy  = np.array([POLICY_JOINTS.index(n) for n in act_jnames])
    policy_to_ctrl  = np.argsort(ctrl_to_policy)

    print("\n[관절 매핑]  policy → mj")
    for pi, name in enumerate(POLICY_JOINTS):
        print(f"  [{pi:2d}] → mj[{policy_to_mj[pi]:2d}]  {name}")

    return policy_to_mj, policy_to_ctrl


def reset_sim(model, data, policy_to_mj):
    mujoco.mj_resetData(model, data)
    for j in range(12):
        data.qpos[7 + policy_to_mj[j]] = DEFAULT_JOINT_POS[j]

    data.qpos[2] = 1.0
    mujoco.mj_forward(model, data)
    foot_names = {"FL_foot","FR_foot","RL_foot","RR_foot","FL","FR","RL","RR"}
    # clearance = geom_center_z - sphere_radius  (bottom of sphere above ground)
    foot_clearances = []
    for i in range(model.ngeom):
        if model.geom(i).name in foot_names:
            center_z = data.geom_xpos[i, 2]
            radius   = model.geom_size[i, 0] if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE else 0.0
            foot_clearances.append(center_z - radius)
    data.qpos[2] = (1.0 - min(foot_clearances)) if foot_clearances else 0.35  # feet just touching ground
    mujoco.mj_forward(model, data)
    print(f"[INFO] initial base z = {data.qpos[2]:.3f} m")


# ─── 관측값 ───────────────────────────────────────────────────────────────────

def compute_obs(data, cmd_vel, last_action, policy_to_mj):
    quat         = data.qpos[3:7]
    lin_vel_body = quat_rotate_inverse(quat, data.qvel[0:3])
    ang_vel_body = data.qvel[3:6].copy()          # MuJoCo: 이미 body frame
    proj_grav    = quat_rotate_inverse(quat, np.array([0.,0.,-1.]))
    cmds         = np.array(cmd_vel, dtype=np.float64)
    q_policy     = data.qpos[7:][policy_to_mj]
    qdot_policy  = data.qvel[6:][policy_to_mj]
    joint_pos_rel = q_policy - DEFAULT_JOINT_POS

    obs = np.concatenate([
        lin_vel_body, ang_vel_body, proj_grav, cmds,
        joint_pos_rel, qdot_policy, last_action,
    ]).astype(np.float32)
    assert obs.shape == (48,), f"obs shape 오류: {obs.shape}"
    return obs


# ─── 액추에이터 ───────────────────────────────────────────────────────────────

def compute_tau_pd(q, qdot, action):
    """
    Isaac Lab DCMotorCfg 정확 재현.

    1) PD 토크 계산
    2) velocity saturation: tau_max = saturation_effort * (1 - qdot/vel_limit)
       (Isaac Lab actuator_pd.py DCMotor._compute_torques 동일 로직)
    3) effort_limit 클리핑
    """
    q_target = DEFAULT_JOINT_POS + ACTION_SCALE * action
    tau = KP * (q_target - q) + KD * (-qdot)

    # velocity-dependent saturation (DCMotor 특성)
    soft_upper = SATURATION_EFFORT * (1.0 - qdot / VELOCITY_LIMIT)
    soft_lower = SATURATION_EFFORT * (-1.0 - qdot / VELOCITY_LIMIT)
    tau = np.clip(tau, soft_lower, soft_upper)

    return np.clip(tau, -EFFORT_LIMIT, EFFORT_LIMIT)


class ElectricMotorState:
    """
    전기모터 ODE 상태 및 적분기.

    MuJoCo에서는 dω/dt를 MuJoCo가 풀어주므로 전기 방정식만 적분:
        dI/dt = (V_cmd - R(t)·I - Ke·ω) / L
        τ     = Kt · I · gr

    고장 모델: t∈[t0, t0+dur] 구간에서 R을 R_nom → R_fault로 선형 증가.
    """

    def __init__(self, n_joints=12, fault_joint=None,
                 fault_R=3.0, fault_start=np.inf, fault_dur=5.0):
        self.I          = np.zeros(n_joints)
        self.R          = np.full(n_joints, R_NOM)
        self.fault_joint = fault_joint
        self.fault_R     = fault_R
        self.fault_start = fault_start
        self.fault_dur   = fault_dur

    def reset(self):
        self.I[:] = 0.0
        self.R[:] = R_NOM

    def _update_R(self, t):
        if self.fault_joint is None:
            return
        t0, t1 = self.fault_start, self.fault_start + self.fault_dur
        if t < t0:
            r = R_NOM
        elif t >= t1:
            r = self.fault_R
        else:
            r = R_NOM + (self.fault_R - R_NOM) * (t - t0) / self.fault_dur
        self.R[self.fault_joint] = r

    def compute_tau(self, q, qdot, action, sim_time, dt):
        """
        1. V_cmd 계산 (PD 역산 → steady-state 전압)
        2. dI/dt ODE 적분 (Euler, sub-step)
        3. τ = Kt · I · gr
        """
        self._update_R(sim_time)

        q_target  = DEFAULT_JOINT_POS + ACTION_SCALE * action
        tau_des   = KP*(q_target-q) + KD*(-qdot)
        soft_upper = SATURATION_EFFORT * (1.0 - qdot / VELOCITY_LIMIT)
        soft_lower = SATURATION_EFFORT * (-1.0 - qdot / VELOCITY_LIMIT)
        tau_des   = np.clip(np.clip(tau_des, soft_lower, soft_upper), -EFFORT_LIMIT, EFFORT_LIMIT)
        I_des     = tau_des / (Kt * GEAR_RATIO)
        omega     = qdot * GEAR_RATIO
        V_cmd     = np.clip(I_des * self.R + Ke * omega, -V_MAX, V_MAX)

        # Euler sub-step: tau_e = L/R ≈ 0.33ms, dt=5ms → 최소 20 sub-step
        n_sub  = max(20, int(dt / (L_MOTOR / R_NOM * 0.3)))
        dt_sub = dt / n_sub
        I = self.I.copy()
        for _ in range(n_sub):
            dI = (V_cmd - self.R * I - Ke * omega) / L_MOTOR
            I  = I + dI * dt_sub
        I_max    = V_MAX / self.R
        self.I   = np.clip(I, -I_max, I_max)

        tau = np.clip(Kt * self.I * GEAR_RATIO, -EFFORT_LIMIT, EFFORT_LIMIT)
        return tau, V_cmd


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy",   required=True)
    parser.add_argument("--scene",    required=True)
    parser.add_argument("--actuator", default="pd", choices=["pd","electric"],
                        help="액추에이터 모드: pd | electric")
    parser.add_argument("--cmd_vx",   type=float, default=1.0)
    parser.add_argument("--cmd_vy",   type=float, default=0.0)
    parser.add_argument("--cmd_yaw",  type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=30.0)
    # 고장 파라미터 (electric 모드 전용)
    parser.add_argument("--fault_joint", type=int,   default=None,
                        help="고장 관절 인덱스 (policy 순서). None=고장 없음")
    parser.add_argument("--fault_R",     type=float, default=3.0,
                        help="고장 시 저항 [Ω]")
    parser.add_argument("--fault_start", type=float, default=5.0,
                        help="고장 시작 시각 [s]")
    parser.add_argument("--fault_dur",   type=float, default=5.0,
                        help="고장 전이 시간 [s]")
    parser.add_argument("--log_data", type=str, default=None,
                        help="데이터 저장 경로 .npz (예: mujoco_data.npz)")
    parser.add_argument("--keep_dof_params", action="store_true", default=False,
                        help="XML의 damping/armature/frictionloss를 0으로 초기화하지 않음")
    args = parser.parse_args()

    cmd_vel = [args.cmd_vx, args.cmd_vy, args.cmd_yaw]

    # ── 정책 로드 ─────────────────────────────────────────────────────────────
    print(f"[INFO] 정책 로드: {args.policy}")
    policy = torch.jit.load(args.policy, map_location="cpu")
    policy.eval()

    # ── MuJoCo 로드 ───────────────────────────────────────────────────────────
    model = mujoco.MjModel.from_xml_path(args.scene)
    data  = mujoco.MjData(model)
    if abs(model.opt.timestep - SIM_DT) > 1e-6:
        print(f"[경고] XML timestep {model.opt.timestep} → {SIM_DT}")
        model.opt.timestep = SIM_DT

    # ── Isaac Lab 파라미터에 맞게 MuJoCo 보정 ────────────────────────────────
    if args.keep_dof_params:
        print("[INFO] keep_dof_params: damping / armature / frictionloss XML 값 유지")
        for i in range(model.njnt):
            if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            dof = model.jnt_dofadr[i]
            print(f"  {model.joint(i).name:20s}  damp={model.dof_damping[dof]:.3f}"
                  f"  arm={model.dof_armature[dof]:.4f}"
                  f"  fric={model.dof_frictionloss[dof]:.3f}")
    else:
        for i in range(model.njnt):
            if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            dof = model.jnt_dofadr[i]
            model.dof_damping[dof]      = 0.0
            model.dof_armature[dof]     = 0.0
            model.dof_frictionloss[dof] = 0.0
        print("[INFO] joint damping / armature / frictionloss → 0 (Isaac Lab 매칭)")

    policy_to_mj, policy_to_ctrl = build_joint_mapping(model)
    reset_sim(model, data, policy_to_mj)

    # ── 전기모터 상태 초기화 ──────────────────────────────────────────────────
    motor = ElectricMotorState(
        fault_joint  = args.fault_joint,
        fault_R      = args.fault_R,
        fault_start  = args.fault_start,
        fault_dur    = args.fault_dur,
    ) if args.actuator == "electric" else None

    last_action = np.zeros(12, dtype=np.float32)

    # 발 geom ID 탐색 (data.contact 기반 contact force)
    foot_geom_ids = []
    for gname in FOOT_GEOM_NAMES:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
        foot_geom_ids.append(gid)
        status = "OK" if gid >= 0 else "NOT FOUND"
        print(f"[INFO] foot geom '{gname}' → id={gid} ({status})")

    # 데이터 로깅 버퍼
    log_buffers = None
    if args.log_data:
        log_buffers = {"time": [], "tau": [], "q": [], "qdot": [],
                       "lin_vel_b": [], "base_height": [], "cmd": [],
                       "foot_contact": [], "foot_force": []}
        print(f"[INFO] 데이터 로깅 활성화 → {args.log_data}")

    print(f"\n[시뮬레이션 시작]  actuator={args.actuator}")
    print(f"  cmd: vx={cmd_vel[0]:.1f}  vy={cmd_vel[1]:.1f}  yaw={cmd_vel[2]:.1f}")
    if motor and args.fault_joint is not None:
        print(f"  고장: joint[{args.fault_joint}]={POLICY_JOINTS[args.fault_joint]}"
              f"  R {R_NOM}→{args.fault_R} Ω  t={args.fault_start}~{args.fault_start+args.fault_dur}s")
    print()

    # ── 로깅 헤더 ─────────────────────────────────────────────────────────────
    if motor:
        print(f"{'t':>6}  {'h':>6}  {'vel_bx':>7}  {'I_mean':>7}  {'I_max':>7}  "
              f"{'V_mean':>7}  {'R_fault':>8}")
        print("-" * 65)
    else:
        print(f"{'t':>6}  {'h':>6}  {'vel_bx':>7}  {'tau_max':>8}")
        print("-" * 40)

    # ── 시뮬레이션 루프 ───────────────────────────────────────────────────────
    # base body ID 찾기 (카메라 추적용)
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    if base_body_id < 0:
        base_body_id = 1  # fallback

    def _save_log():
        if log_buffers is None or not args.log_data or len(log_buffers["time"]) == 0:
            return
        np.savez(
            args.log_data,
            time=np.array(log_buffers["time"]),
            tau=np.array(log_buffers["tau"]),
            q=np.array(log_buffers["q"]),
            qdot=np.array(log_buffers["qdot"]),
            lin_vel_b=np.array(log_buffers["lin_vel_b"]),
            base_height=np.array(log_buffers["base_height"]),
            cmd=np.array(log_buffers["cmd"]),
            joint_names=np.array(POLICY_JOINTS),
            foot_contact=np.array(log_buffers["foot_contact"]),
            foot_force=np.array(log_buffers["foot_force"]),
            foot_names=np.array(FOOT_NAMES),
            source="mujoco",
        )
        print(f"[INFO] 데이터 저장 완료 → {os.path.abspath(args.log_data)}  ({len(log_buffers['time'])} steps)")

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # 카메라를 로봇 base에 고정 추적
            viewer.cam.type        = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = base_body_id
            viewer.cam.distance    = 2.5   # 로봇으로부터 거리 [m]
            viewer.cam.elevation   = -20   # 내려다보는 각도 [deg]
            viewer.cam.azimuth     = 180   # 뒤에서 바라보는 방향

            step = 0

            while viewer.is_running() and data.time < args.duration:
                t_wall = time.perf_counter()

                # 관측 & 정책
                obs = compute_obs(data, cmd_vel, last_action, policy_to_mj)
                with torch.no_grad():
                    action = policy(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()
                last_action = action.copy()

                # Decimation 루프: 매 physics step마다 액추에이터 재계산
                for _ in range(DECIMATION):
                    q_now    = data.qpos[7:][policy_to_mj]
                    qdot_now = data.qvel[6:][policy_to_mj]

                    if motor:
                        tau, V_cmd = motor.compute_tau(q_now, qdot_now, action,
                                                       data.time, SIM_DT)
                    else:
                        tau = compute_tau_pd(q_now, qdot_now, action)

                    data.ctrl[policy_to_ctrl] = tau
                    mujoco.mj_step(model, data)

                step += 1
                viewer.sync()

                # 데이터 수집 (policy step마다)
                if log_buffers is not None:
                    quat     = data.qpos[3:7]
                    vb       = quat_rotate_inverse(quat, data.qvel[0:3])
                    q_now    = data.qpos[7:][policy_to_mj]
                    qdot_now = data.qvel[6:][policy_to_mj]
                    log_buffers["time"].append(data.time)
                    log_buffers["tau"].append(tau.copy())
                    log_buffers["q"].append(q_now.copy())
                    log_buffers["qdot"].append(qdot_now.copy())
                    log_buffers["lin_vel_b"].append(vb.copy())
                    log_buffers["base_height"].append(data.qpos[2])
                    log_buffers["cmd"].append(np.array(cmd_vel, dtype=np.float32))
                    # 발 contact force: data.contact 순회 → mj_contactForce
                    foot_force_arr = np.zeros(4, dtype=np.float32)
                    _wrench = np.zeros(6)
                    for ci in range(data.ncon):
                        c = data.contact[ci]
                        for fi, gid in enumerate(foot_geom_ids):
                            if gid >= 0 and (c.geom1 == gid or c.geom2 == gid):
                                mujoco.mj_contactForce(model, data, ci, _wrench)
                                foot_force_arr[fi] += np.linalg.norm(_wrench[:3])
                    log_buffers["foot_force"].append(foot_force_arr.copy())
                    log_buffers["foot_contact"].append((foot_force_arr > 1.0).astype(np.float32))

                # 로깅 (1초마다)
                if step % 50 == 0:
                    quat = data.qpos[3:7]
                    vb   = quat_rotate_inverse(quat, data.qvel[0:3])
                    h    = data.qpos[2]
                    if motor:
                        R_f = motor.R[args.fault_joint] if args.fault_joint is not None else R_NOM
                        print(f"{data.time:6.2f}  {h:6.3f}  {vb[0]:+7.3f}  "
                              f"{np.abs(motor.I).mean():7.3f}  {np.abs(motor.I).max():7.3f}  "
                              f"{np.abs(V_cmd).mean():7.3f}  {R_f:8.4f}")
                    else:
                        print(f"{data.time:6.2f}  {h:6.3f}  {vb[0]:+7.3f}  {np.abs(tau).max():8.3f}")

                elapsed = time.perf_counter() - t_wall
                sleep_t = SIM_DT * DECIMATION - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n[INFO] 사용자 중단.")
    finally:
        _save_log()

    print(f"\n[완료] {step} steps  t={data.time:.1f}s")


if __name__ == "__main__":
    main()
