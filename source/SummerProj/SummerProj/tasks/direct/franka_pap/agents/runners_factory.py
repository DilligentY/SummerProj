import yaml
from typing import Dict, Tuple
from isaaclab.envs import DirectRLEnv

# Custom Model (Relative Path)
from ..models.custom_net import FrankaGaussianPolicy, FrankaValue

from skrl.agents.torch.ppo import PPO
from skrl.memories.torch import RandomMemory
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

def create_ppo_runner(env: DirectRLEnv, 
                      memory: Dict,
                      agent: Dict, 
                      trainer: Dict, 
                      seed: int, 
                      **kwargs) -> Tuple[SequentialTrainer, PPO]:
    """
        커스텀 PPO 에이전트를 생성, 설정 및 반환하는 함수
        
            1. Model 생성 : Customized Neural Network Model
            2. Buffer 생성 : Memory Buffer for On-policy algorithm
            3. Agent 생성 : PPO Agent for learning process
            4. Trainer 생성 : Sequential Trainer for learning process
    """
    print("[RunnerFactory] Creating a custom PPO agent and trainer ...")
    # 팩토리 함수 내부에서 시드 설정
    set_seed(seed)

    # 1. 커스텀 모델 분리 생성
    policy = FrankaGaussianPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device
    )
    value = FrankaValue(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )
    models = {"policy": policy, "value": value}
    
    # 2. 메모리 생성
    memory_size = agent.get("rollouts")
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=env.device)

    # 3. PPO 설정을 위한 딕셔너리 전처리
    ppo_cfg = agent
    
    # kwargs가 None일 경우 빈 딕셔너리로 교체
    if ppo_cfg.get("state_preprocessor_kwargs", None) is None:
        ppo_cfg["state_preprocessor_kwargs"] = {}
    if ppo_cfg.get("value_preprocessor_kwargs", None) is None:
        ppo_cfg["value_preprocessor_kwargs"] = {}

    # 문자열을 실제 클래스 객체로 변환
    if ppo_cfg.get("learning_rate_scheduler", None) == "KLAdaptiveLR":
        ppo_cfg["learning_rate_scheduler"] = KLAdaptiveLR
    if ppo_cfg.get("state_preprocessor", None) == "RunningStandardScaler":
        ppo_cfg["state_preprocessor"] = RunningStandardScaler
        ppo_cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": env.device}
    if ppo_cfg.get("value_preprocessor", None) == "RunningStandardScaler":
        ppo_cfg["value_preprocessor"] = RunningStandardScaler
        ppo_cfg["value_preprocessor_kwargs"] = {"size": 1, "device": env.device}
    
    # 4. 최종 PPO 에이전트 생성
    agent = PPO(models=models,
                memory=memory,
                cfg=ppo_cfg,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=env.device)
    
    # 5. 트레이너 생성
    trainer_cfg = trainer
    trainer = SequentialTrainer(env=env, agents=agent, cfg=trainer_cfg)

    print("[RunnerFactory] Custom trainer and agent created successfully.")
    return trainer, agent