"""
MuJoCo play script - Unitree Go2 Isaac Lab 정책 재현

PD 토크 제어 (Isaac Lab DCMotor 재현):
  τ = Kp·(q_target - q) + Kd·(-qdot)
  velocity saturation 및 effort_limit 클리핑 포함

실행 예시:
  python play_mujoco.py --policy policy.pt --scene go2_scene.xml
  python play_mujoco.py --policy policy.pt --scene go2_scene.xml --cmd_vx 1.5 --duration 60
  python play_mujoco.py --policy policy.pt --scene go2_scene.xml --log_data mujoco_data.npz
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
SIM_DT            = 0.005  # s  (velocity_env_cfg.py sim.dt와 동일)
DECIMATION        = 4      # → policy 50Hz
ACTION_SCALE      = 0.25

# ─── 관절 정의 ────────────────────────────────────────────────────────────────
POLICY_JOINTS = [
    "FL_hip_joint",   "FR_hip_joint",   "RL_hip_joint",   "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint",  "FR_calf_joint",  "RL_calf_joint",  "RR_calf_joint",
]

FOOT_GEOM_NAMES = ["FL", "FR", "RL", "RR"]
FOOT_NAMES      = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

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

    act_jnames     = [model.joint(model.actuator_trnid[i, 0]).name for i in range(model.nu)]
    ctrl_to_policy = np.array([POLICY_JOINTS.index(n) for n in act_jnames])
    policy_to_ctrl = np.argsort(ctrl_to_policy)

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
    foot_names = {"FL_foot", "FR_foot", "RL_foot", "RR_foot", "FL", "FR", "RL", "RR"}
    foot_clearances = []
    for i in range(model.ngeom):
        if model.geom(i).name in foot_names:
            center_z = data.geom_xpos[i, 2]
            radius   = model.geom_size[i, 0] if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE else 0.0
            foot_clearances.append(center_z - radius)
    data.qpos[2] = (1.0 - min(foot_clearances)) if foot_clearances else 0.35
    mujoco.mj_forward(model, data)
    print(f"[INFO] initial base z = {data.qpos[2]:.3f} m")


# ─── 관측값 ───────────────────────────────────────────────────────────────────

def compute_obs(data, cmd_vel, last_action, policy_to_mj):
    quat          = data.qpos[3:7]
    lin_vel_body  = quat_rotate_inverse(quat, data.qvel[0:3])
    ang_vel_body  = data.qvel[3:6].copy()
    proj_grav     = quat_rotate_inverse(quat, np.array([0., 0., -1.]))
    cmds          = np.array(cmd_vel, dtype=np.float64)
    q_policy      = data.qpos[7:][policy_to_mj]
    qdot_policy   = data.qvel[6:][policy_to_mj]
    joint_pos_rel = q_policy - DEFAULT_JOINT_POS

    obs = np.concatenate([
        lin_vel_body, ang_vel_body, proj_grav, cmds,
        joint_pos_rel, qdot_policy, last_action,
    ]).astype(np.float32)
    assert obs.shape == (48,), f"obs shape 오류: {obs.shape}"
    return obs


# ─── 액추에이터 (PD, Isaac Lab DCMotor 재현) ──────────────────────────────────

def compute_tau_pd(q, qdot, action):
    """
    Isaac Lab DCMotorCfg 정확 재현.
    1) PD 토크 계산
    2) velocity saturation: tau_max = saturation_effort * (1 - qdot/vel_limit)
    3) effort_limit 클리핑
    """
    q_target   = DEFAULT_JOINT_POS + ACTION_SCALE * action
    tau        = KP * (q_target - q) + KD * (-qdot)
    soft_upper = SATURATION_EFFORT * (1.0 - qdot / VELOCITY_LIMIT)
    soft_lower = SATURATION_EFFORT * (-1.0 - qdot / VELOCITY_LIMIT)
    tau        = np.clip(tau, soft_lower, soft_upper)
    return np.clip(tau, -EFFORT_LIMIT, EFFORT_LIMIT)


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy",   required=True)
    parser.add_argument("--scene",    required=True)
    parser.add_argument("--cmd_vx",   type=float, default=1.0)
    parser.add_argument("--cmd_vy",   type=float, default=0.0)
    parser.add_argument("--cmd_yaw",  type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--log_data", type=str,   default=None,
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

    last_action = np.zeros(12, dtype=np.float32)

    # ── 발 geom ID 탐색 ───────────────────────────────────────────────────────
    foot_geom_ids = []
    for gname in FOOT_GEOM_NAMES:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
        foot_geom_ids.append(gid)
        status = "OK" if gid >= 0 else "NOT FOUND"
        print(f"[INFO] foot geom '{gname}' → id={gid} ({status})")

    # ── 데이터 로깅 버퍼 ──────────────────────────────────────────────────────
    log_buffers = None
    if args.log_data:
        log_buffers = {"time": [], "tau": [], "q": [], "qdot": [],
                       "lin_vel_b": [], "base_height": [], "cmd": [],
                       "foot_contact": [], "foot_force": []}
        print(f"[INFO] 데이터 로깅 활성화 → {args.log_data}")

    print(f"\n[시뮬레이션 시작]")
    print(f"  cmd: vx={cmd_vel[0]:.1f}  vy={cmd_vel[1]:.1f}  yaw={cmd_vel[2]:.1f}")
    print(f"\n{'t':>6}  {'h':>6}  {'vel_bx':>7}  {'tau_max':>8}")
    print("-" * 40)

    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    if base_body_id < 0:
        base_body_id = 1

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
            viewer.cam.type        = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = base_body_id
            viewer.cam.distance    = 2.5
            viewer.cam.elevation   = -20
            viewer.cam.azimuth     = 180

            step = 0

            while viewer.is_running() and data.time < args.duration:
                t_wall = time.perf_counter()

                # 관측 & 정책
                obs = compute_obs(data, cmd_vel, last_action, policy_to_mj)
                with torch.no_grad():
                    action = policy(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()
                last_action = action.copy()

                # Decimation 루프
                for _ in range(DECIMATION):
                    q_now    = data.qpos[7:][policy_to_mj]
                    qdot_now = data.qvel[6:][policy_to_mj]
                    tau      = compute_tau_pd(q_now, qdot_now, action)
                    data.ctrl[policy_to_ctrl] = tau
                    mujoco.mj_step(model, data)

                step += 1
                viewer.sync()

                # 데이터 수집
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
