# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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
# fixed velocity command (sim-to-sim comparison)
parser.add_argument("--cmd_vx",  type=float, default=None, help="Fix forward velocity command [m/s]")
parser.add_argument("--cmd_vy",  type=float, default=None, help="Fix lateral velocity command [m/s]")
parser.add_argument("--cmd_yaw", type=float, default=None, help="Fix yaw-rate command [rad/s]")
# initial base height for sim-to-sim (read from MuJoCo "[INFO] initial base z = X.XXX m")
parser.add_argument("--sim2sim_init_z", type=float, default=0.4,
                    help="Initial base z [m] for sim-to-sim (match MuJoCo reset height, default=0.4)")
# data logging for sim-to-sim comparison
parser.add_argument("--log_data", type=str, default=None,
                    help="Save logged data to this .npz path (e.g. isaaclab_data.npz)")
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
import numpy as np
import torch
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
# import sys
# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # fix velocity commands for sim-to-sim comparison
    if args_cli.cmd_vx is not None or args_cli.cmd_vy is not None or args_cli.cmd_yaw is not None:
        vx  = args_cli.cmd_vx  if args_cli.cmd_vx  is not None else 0.0
        vy  = args_cli.cmd_vy  if args_cli.cmd_vy  is not None else 0.0
        yaw = args_cli.cmd_yaw if args_cli.cmd_yaw is not None else 0.0
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (vx,  vx)
        env_cfg.commands.base_velocity.ranges.lin_vel_y = (vy,  vy)
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (yaw, yaw)
        env_cfg.commands.base_velocity.heading_command  = False  # use ang_vel_z directly
        env_cfg.commands.base_velocity.rel_standing_envs = 0.0   # no standing envs
        print(f"[INFO] Fixed velocity command: vx={vx:.2f}  vy={vy:.2f}  yaw={yaw:.2f}")

        # sim-to-sim: set initial base height to match MuJoCo ground-contact height
        # MuJoCo reset_sim prints "[INFO] initial base z = X.XXX m" — use that value here
        env_cfg.scene.robot.init_state.pos = (0.0, 0.0, args_cli.sim2sim_init_z)
        # fix yaw/xy range to 0 so starting pose is deterministic
        env_cfg.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0),
        }
        # disable startup DR events (mass, friction, motor — absent in MuJoCo)
        env_cfg.events.physics_material = None
        env_cfg.events.add_base_mass    = None
        env_cfg.events.motor_strength   = None
        print(f"[INFO] Sim-to-sim mode: init z={args_cli.sim2sim_init_z:.4f} m, all DR disabled")

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # print(f"[INFO] robot prim_path: {env_cfg.scene.robot.prim_path}")

    # specify directory for logging experiments
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

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    inner_env = env.unwrapped  # ManagerBasedRLEnv reference (stable before wrapping)
    robot = inner_env.scene["robot"]
    
    # 발 contact sensor 탐색
    _FOOT_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    try:
        contact_sensor = inner_env.scene["contact_forces"]
        _sensor_bodies = list(contact_sensor.body_names)
        # "_foot" 포함 이름 중 FL/FR/RL/RR 순서로 정렬
        _foot_sensor_idx = []
        for fname in _FOOT_NAMES:
            matches = [i for i, n in enumerate(_sensor_bodies) if fname in n]
            _foot_sensor_idx.append(matches[0] if matches else -1)
        _has_contact = all(i >= 0 for i in _foot_sensor_idx)
        print(f"[INFO] Contact sensor 발 인덱스: {dict(zip(_FOOT_NAMES, _foot_sensor_idx))}")
        print("[INFO] Contact sensor body list:")                                                       
        for i, name in enumerate(_sensor_bodies):                                                       
            print(f"  [{i:2d}] {name}")
        inertias = robot.root_physx_view.get_inertias()[0]  # (num_bodies, 9)                           
        masses   = robot.root_physx_view.get_masses()[0]    # (num_bodies,)                             
        body_names = list(robot.body_names)                                                             
        for name, m in zip(body_names, masses):                                                         
            print(f"  {name}: {m:.4f} kg")
    except (KeyError, AttributeError):
        _has_contact = False
        print("[INFO] Contact sensor 없음 — foot_contact/force 미수집")

    for actuator in robot.actuators.values():
        if hasattr(actuator, 'inject_physx_view'):
            actuator.inject_physx_view(robot.root_physx_view)
            print(f"[INFO] Injected physx_view into {actuator.__class__.__name__}")

    # print joint armature values from PhysX (rotor inertia check)
    armatures = robot.root_physx_view.get_dof_armatures()[0].tolist()
    print("[INFO] Joint armatures (rotor inertia) from PhysX:")
    for name, val in zip(robot.data.joint_names, armatures):
        print(f"  {name}: {val:.6f} kg·m²")

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
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

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # data logging setup
    _saved_joint_names = []  # closure용 — log_buffers가 None 되기 전에 캡처
    log_buffers = None
    sim_time = 0.0
    if args_cli.log_data:
        log_buffers = {"time": [], "tau": [], "q": [], "qdot": [],
                       "lin_vel_b": [], "base_height": [], "cmd": [],
                       "foot_contact": [], "foot_force": []}
        _saved_joint_names[:] = list(robot.data.joint_names)
        print(f"[INFO] Data logging enabled → {args_cli.log_data}")
        print(f"[INFO] Joint order: {_saved_joint_names}")

    def _save_log():
        if log_buffers is None or not args_cli.log_data or len(log_buffers["time"]) == 0:
            return
        names = _saved_joint_names if _saved_joint_names else []
        save_path = args_cli.log_data
        np.savez(
            save_path,
            time=np.array(log_buffers["time"]),
            tau=np.array(log_buffers["tau"]),
            q=np.array(log_buffers["q"]),
            qdot=np.array(log_buffers["qdot"]),
            lin_vel_b=np.array(log_buffers["lin_vel_b"]),
            base_height=np.array(log_buffers["base_height"]),
            cmd=np.array(log_buffers["cmd"]),
            joint_names=np.array(names),
            foot_contact=np.array(log_buffers["foot_contact"]),
            foot_force=np.array(log_buffers["foot_force"]),
            foot_names=np.array(_FOOT_NAMES),
            source="isaaclab",
        )
        print(f"[INFO] Data saved → {os.path.abspath(save_path)}  ({len(log_buffers['time'])} steps)")

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)

        # collect data for sim-to-sim comparison (env 0 only)
        if log_buffers is not None:
            sim_time += dt
            # joint torques: try applied_torque first, fall back to zeros
            try:
                tau = robot.data.applied_torque[0].cpu().numpy()
            except AttributeError:
                tau = np.zeros(robot.data.joint_pos.shape[-1])
            log_buffers["time"].append(sim_time)
            log_buffers["tau"].append(tau.copy())
            log_buffers["q"].append(robot.data.joint_pos[0].cpu().numpy().copy())
            log_buffers["qdot"].append(robot.data.joint_vel[0].cpu().numpy().copy())
            log_buffers["lin_vel_b"].append(robot.data.root_lin_vel_b[0].cpu().numpy().copy())
            log_buffers["base_height"].append(robot.data.root_pos_w[0, 2].cpu().item())
            # command: [lin_vel_x, lin_vel_y, ang_vel_z]
            cmd_raw = inner_env.command_manager.get_command("base_velocity")[0].cpu().numpy()
            log_buffers["cmd"].append(cmd_raw[:3].copy())  # take first 3 (vx, vy, yaw)
            # 발 contact force
            if _has_contact:
                foot_f = contact_sensor.data.net_forces_w[0, _foot_sensor_idx, :]  # (4, 3)
                foot_force_mag = torch.norm(foot_f, dim=-1).cpu().numpy().astype(np.float32)
            else:
                foot_force_mag = np.zeros(4, dtype=np.float32)
            log_buffers["foot_force"].append(foot_force_mag.copy())
            log_buffers["foot_contact"].append((foot_force_mag > 1.0).astype(np.float32))

            # episode done for env 0 → save and stop logging
            if dones[0]:
                _save_log()
                log_buffers = None  # 이후 수집 중단 (중복 저장 방지)
                print("[INFO] Episode 0 done. Data saved. Continuing simulation...")

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
