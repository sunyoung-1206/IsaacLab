"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--state", type=str, default="normal", choices=["normal", "faulty"], help="데이터 상태 구분 (normal/faulty)")
# ===== 명령 모드 추가 =====
parser.add_argument(
    "--command_mode",
    type=str,
    default="random",
    choices=["random", "fixed"],
    help="명령 모드: 'random'은 기존 랜덤 방식, 'fixed'는 고정 에피소드 시퀀스 사용",
)
parser.add_argument(
    "--episode_steps",
    type=int,
    default=1500,
    help="[fixed 모드] 에피소드당 스텝 수 (500 스텝이면 약 10초)",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch
import numpy as np
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_assets.actuators.electric_motor import ElectricMotor
# PLACEHOLDER: Extension template (do not remove this comment)

# ===== 고정 에피소드 시퀀스 정의 =====
# (vx, vy, yaw) 순서, 에피소드 0→1→2 순차 실행
FIXED_EPISODE_COMMANDS = [
    (0.5, 0.0, 0.0),   # episode 0
    (1.0, 0.0, 0.0),   # episode 1
    (2.0, 0.0, 0.0),   # episode 2
]


def make_empty_log():
    return {
        "cmd_vel": [],
        "actions": [],
        "joint_pos": [],
        "joint_vel": [],
        "joint_torque": [],
        "joint_power": [],
        "base_pos": [],
        "base_quat": [],
        "base_lin_vel": [],
        "base_ang_vel": [],
        "timestamp": [],
        "motor_current": [],
        "motor_voltage": [],
        "motor_temperature": [],
    }


def save_episode(log_data, episode_dir, env_idx):
    save_path = os.path.join(episode_dir, f"env{env_idx:02d}.npz")
    save_dict = {k: np.array(v) for k, v in log_data.items() if len(v) > 0}
    # timestamp를 에피소드 시작 기준으로 0부터 시작하도록 normalize
    if "timestamp" in save_dict and len(save_dict["timestamp"]) > 0:
        save_dict["timestamp"] = save_dict["timestamp"] - save_dict["timestamp"][0]
    np.savez(save_path, **save_dict)
    print(f"      Saved: {os.path.basename(episode_dir)}/env{env_idx:02d}.npz "
          f"({len(log_data['timestamp'])} steps)")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.resampling_time_range = (9999.0, 9999.0)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.state == "faulty":
        robot = env.unwrapped.scene["robot"]
        for actuator in robot.actuators.values():
            if isinstance(actuator, ElectricMotor):
                actuator.cfg.faulty = True

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    timestep = 0

    num_envs = env.unwrapped.num_envs
    log_data_all_envs = [make_empty_log() for _ in range(num_envs)]
    command_episodes = [0] * num_envs
    prev_commands = [None] * num_envs
    current_episode_dir = None

    # ===== fixed 모드 전용 상태 변수 =====
    fixed_mode = args_cli.command_mode == "fixed"
    fixed_episode_idx = 0           # 현재 진행 중인 에피소드 인덱스
    fixed_step_in_episode = 0       # 현재 에피소드 내 경과 스텝
    fixed_total_episodes = len(FIXED_EPISODE_COMMANDS)
    fixed_episode_steps = args_cli.episode_steps
    fixed_done = False              # 모든 에피소드 완료 플래그

    blacklisted_envs = set()

    if fixed_mode:
        print(f"[INFO] Command mode: FIXED")
        print(f"[INFO] Episodes: {fixed_total_episodes}, Steps per episode: {fixed_episode_steps}")
        for i, cmd in enumerate(FIXED_EPISODE_COMMANDS):
            print(f"       Episode {i}: vx={cmd[0]}, vy={cmd[1]}, yaw={cmd[2]}")
    else:
        print(f"[INFO] Command mode: RANDOM (original behavior)")

    print(f"[INFO] Starting data logging for {num_envs} environments...")

    while simulation_app.is_running():
        if fixed_mode and fixed_done:
            break

        start_time = time.time()
        with torch.inference_mode():
            if hasattr(env.unwrapped, 'command_manager'):
                all_commands = env.unwrapped.command_manager.get_command("base_velocity")

                if fixed_mode:
                    # ===== fixed 모드: 현재 에피소드 명령을 모든 env에 강제 주입 =====
                    vx, vy, yaw = FIXED_EPISODE_COMMANDS[fixed_episode_idx]
                    fixed_cmd = torch.tensor([vx, vy, yaw], device=all_commands.device)
                    all_commands[:] = fixed_cmd.unsqueeze(0).expand(num_envs, -1)
                else:
                    # ===== random 모드: 기존 방식 (env[0] 기준 브로드캐스트) =====
                    reference_command = all_commands[0:1]
                    all_commands[:] = reference_command.repeat(num_envs, 1)

            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)

            done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
            for idx in done_indices:
                blacklisted_envs.add(idx.item())

            robot = env.unwrapped.scene["robot"]
            current_commands = all_commands.cpu().numpy()

            for env_idx in range(num_envs):
                if env_idx in blacklisted_envs:
                    continue
                log_data = log_data_all_envs[env_idx]

                log_data["actions"].append(actions[env_idx].cpu().numpy())
                log_data["joint_pos"].append(robot.data.joint_pos[env_idx].cpu().numpy())
                log_data["joint_vel"].append(robot.data.joint_vel[env_idx].cpu().numpy())

                torque = robot.data.applied_torque[env_idx]
                vel = robot.data.joint_vel[env_idx]
                log_data["joint_torque"].append(torque.cpu().numpy())

                for actuator in robot.actuators.values():
                    if hasattr(actuator, "current"):
                        log_data["motor_current"].append(actuator.current[env_idx].cpu().numpy())
                        log_data["motor_voltage"].append(actuator.voltage[env_idx].cpu().numpy())
                        log_data["motor_temperature"].append(actuator.temperature[env_idx].cpu().numpy())

                log_data["joint_power"].append((torque * vel).cpu().numpy())
                log_data["base_pos"].append(robot.data.root_pos_w[env_idx].cpu().numpy())
                log_data["base_quat"].append(robot.data.root_quat_w[env_idx].cpu().numpy())
                log_data["base_lin_vel"].append(robot.data.root_lin_vel_b[env_idx].cpu().numpy())
                log_data["base_ang_vel"].append(robot.data.root_ang_vel_b[env_idx].cpu().numpy())

                current_command = current_commands[env_idx]
                log_data["cmd_vel"].append(current_command)
                log_data["timestamp"].append(timestep * dt)

                # ===== 에피소드 전환 감지 및 저장 =====
                if fixed_mode:
                    # fixed 모드: 스텝 카운트 기반 에피소드 전환 (env_idx==0 기준)
                    pass  # 아래에서 일괄 처리
                else:
                    # random 모드: 기존 명령 변경 감지 방식
                    if prev_commands[env_idx] is not None:
                        command_changed = np.linalg.norm(current_command - prev_commands[env_idx]) > 0.1

                        if command_changed and len(log_data["timestamp"]) > 10:
                            if env_idx == 0:
                                state_dir = os.path.join(log_dir, args_cli.state)
                                episode_num = command_episodes[0]
                                cmd = prev_commands[0]
                                episode_folder_name = (f"episode_{episode_num:03d}_"
                                                       f"vx{cmd[0]:.2f}_vy{cmd[1]:.2f}_yaw{cmd[2]:.2f}")
                                current_episode_dir = os.path.join(state_dir, episode_folder_name)
                                os.makedirs(current_episode_dir, exist_ok=True)
                                print(f"\n[INFO] Saving {args_cli.state} data to: {current_episode_dir}")
                                print(f"\n[INFO] Command changed from {cmd} to {current_command}")

                            save_episode(log_data_all_envs[env_idx], current_episode_dir, env_idx)
                            command_episodes[env_idx] += 1
                            log_data_all_envs[env_idx] = make_empty_log()

                prev_commands[env_idx] = current_command.copy()

            # ===== fixed 모드: 에피소드 스텝 카운트 관리 =====
            if fixed_mode:
                fixed_step_in_episode += 1

                if fixed_step_in_episode >= fixed_episode_steps:
                    # 현재 에피소드 데이터 저장
                    vx, vy, yaw = FIXED_EPISODE_COMMANDS[fixed_episode_idx]
                    state_dir = os.path.join(log_dir, args_cli.state)
                    episode_folder_name = (f"episode_{fixed_episode_idx:03d}_"
                                           f"vx{vx:.2f}_vy{vy:.2f}_yaw{yaw:.2f}")
                    current_episode_dir = os.path.join(state_dir, episode_folder_name)
                    os.makedirs(current_episode_dir, exist_ok=True)

                    print(f"\n[INFO] Episode {fixed_episode_idx} complete → saving to: {episode_folder_name}")
                    for env_idx in range(num_envs):
                        if env_idx in blacklisted_envs:
                            print(f"  [SKIP] env{env_idx} blacklisted (fell down)")
                            continue
                        save_episode(log_data_all_envs[env_idx], current_episode_dir, env_idx)

                    # 다음 에피소드 준비
                    fixed_episode_idx += 1
                    fixed_step_in_episode = 0
                    log_data_all_envs = [make_empty_log() for _ in range(num_envs)]

                    if fixed_episode_idx >= fixed_total_episodes:
                        print(f"\n[INFO] All {fixed_total_episodes} episodes complete. Exiting.")
                        fixed_done = True

                    else:
                        obs, _ = env.reset()

        timestep += 1
        if args_cli.video and timestep >= args_cli.video_length:
            break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()