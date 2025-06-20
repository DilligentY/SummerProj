# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import numpy as np

from isaaclab.app import AppLauncher
# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=3, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Franka-Reach-Direct-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)

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

import gymnasium as gym
import os
import random
from datetime import datetime

import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.2"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import SummerProj.tasks  # noqa: F401


# --- [수정 1] ---
# 우리가 만든 커스텀 모델 클래스를 임포트합니다.
# 이 경로는 사용자님의 프로젝트 구조에 맞게 정확해야 합니다.
try:
    from SummerProj.tasks.direct.franka_pap.agents.custom_net import FrankaGaussianPolicy, FrankaValue
    # from source.SummerProj.SummerProj.tasks.direct.franka_pap.agents.custom_net import FrankaCustomActorCritic
except ImportError:
    # 이 부분은 코드가 다른 위치에 있을 경우를 대비한 예외처리입니다. 경로를 확인하세요.
    print("Warning: Could not import FrankaCustomActorCritic. Make sure the path is correct.")
    FrankaCustomActorCritic = None
# --------------------

# --- [수정 1-1] ---
# skrl에서 PPO 에이전트와 Memory 클래스를 직접 임포트합니다.
from skrl.agents.torch.ppo import PPO
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.resources.preprocessors.torch import RunningStandardScaler
# --------------------



# config shortcuts
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    print(f"Exact experiment name requested from command line {log_dir}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # get checkpoint path (to resume training)
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`


    # # --- [최종 해결책: PPO 에이전트를 직접 생성 후 Runner에 전달] ---
    print("[INFO] Bypassing skrl Runner's automatic agent generation to use a custom model.")
    # 1. 정책 모델을 먼저 생성합니다.
    policy = FrankaGaussianPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
        # ... (필요한 경우 YAML에서 파라미터 로드)
    )

    # 2. 가치 모델을 생성합니다.
    #    이때, 정책 모델의 인코더를 전달하여 가중치를 공유(weight sharing)합니다.
    value = FrankaValue(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )
    models = {"policy": policy, "value": value}
    
    # 2. skrl의 메모리 인스턴스 생성
    # agent_cfg에서 rollouts 수를 가져와 memory_size로 사용
    memory_size = agent_cfg.get("agent", {}).get("rollouts")
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=env.device)

    # 3. PPO 에이전트 생성을 위해 설정(cfg) 딕셔너리를 전처리합니다.
    ppo_cfg = agent_cfg.get("agent", {})
    
    # 3.1. 문자열로 된 구성요소를 실제 skrl 클래스 객체로 변환
    print("[INFO] Manually processing agent configuration...")
    if ppo_cfg.get("state_preprocessor_kwargs", None) is None:
        ppo_cfg["state_preprocessor_kwargs"] = {}
    
    if ppo_cfg.get("value_preprocessor_kwargs", None) is None:
        ppo_cfg["value_preprocessor_kwargs"] = {}

    # 3.2. 문자열로 된 구성요소를 실제 skrl 클래스 객체로 변환 (이전 단계에서 수정한 내용)
    if ppo_cfg.get("learning_rate_scheduler", None) == "KLAdaptiveLR":
        ppo_cfg["learning_rate_scheduler"] = KLAdaptiveLR
    
    if ppo_cfg.get("state_preprocessor", None) == "RunningStandardScaler":
        ppo_cfg["state_preprocessor"] = RunningStandardScaler
        ppo_cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": env.device}

    if ppo_cfg.get("value_preprocessor", None) == "RunningStandardScaler":
        ppo_cfg["value_preprocessor"] = RunningStandardScaler
        ppo_cfg["value_preprocessor_kwargs"] = {"size": 1, "device": env.device}
    print(f"[INFO] Set preprocessor sizes: state={env.observation_space.shape}, value=1")

    # 3.3. Runner가 사용할 설정은 미리 제거
    ppo_cfg.pop("experiment", None)
    
    # 4. 이제 모든 내용이 올바르게 변환된 cfg를 사용하여 PPO 에이전트를 직접 생성합니다.
    agent = PPO(models=models,
                memory=memory,
                cfg=ppo_cfg, # 전처리된 ppo_cfg 전달
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=env.device)
    

    # 5. Trainer를 직접 생성합니다.
    #    cfg에는 agent_cfg의 'trainer' 하위 딕셔너리를 전달합니다.
    trainer_cfg = agent_cfg.get("trainer", {})
    trainer = SequentialTrainer(env=env, agents=agent, cfg=trainer_cfg)

    # 6. Checkpoint 로드 (이전에는 Runner가 하던 일)
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        agent.load(resume_path) # runner.agent.load가 아닌 agent.load
    
    trainer.train()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
