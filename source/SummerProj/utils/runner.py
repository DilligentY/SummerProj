from typing import Any, Mapping
import copy

# hydra.utils.instantiate를 사용하기 위해 임포트
import hydra
from omegaconf import DictConfig

# skrl의 원래 Runner와 필요한 클래스들을 임포트
from skrl import logger
from skrl.utils.runner.torch import Runner
from skrl.models.torch import Model


class AISLRunner(Runner):

    def _generate_models(self, env, cfg: Mapping[str, Any]) -> Mapping[str, Mapping[str, Model]]:
        multi_agent = False
        device = env.device
        possible_agents = ["agent"]
        observation_spaces = {"agent": env.observation_space}
        action_spaces = {"agent": env.action_space}

        models = {}
        for agent_id in possible_agents:
            models[agent_id] = {}
            models_cfg = copy.deepcopy(cfg.get("models", {}))
            if not models_cfg:
                raise ValueError("No 'models' are defined in cfg")
            
            try:
                separate = models_cfg["separate"]
                del models_cfg["separate"]
            except KeyError:
                separate = True
                logger.warning("No 'separate' field defined in 'models' cfg. Defining it as True by default")
            # non-shared models
            if separate:
                for role in models_cfg:
                    model_config = models_cfg[role]
            
                    if "_target_" in model_config:
                        print(f"[CustomRunner] Instantiating model for role '{role}' via Hydra _target_...")
                        # 커스텀 신호인 _target_이 있다면, Hydra를 사용해 모델을 직접 생성
                        # 모델의 __init__에 필요한 기본 인자들을 추가
                        model_config["observation_space"] = observation_spaces[agent_id]
                        model_config["action_space"] = action_spaces[agent_id]
                        model_config["device"] = device
                        
                        # hydra.utils.instantiate는 DictConfig를 받으므로 변환
                        if not isinstance(model_config, DictConfig):
                            model_config = hydra.utils.instantiate({"_target_": "omegaconf.OmegaConf.create", "_val": model_config})

                        models[agent_id][role] = hydra.utils.instantiate(model_config)
                    else:
                        # _target_이 없다면, 원래 SKRL에서 제공하는 Runner 클래스 그대로 사용
                        print(f"[CustomRunner] Instantiating model for role '{role}' via skrl's default method...")
                        model_class_str = model_config.get("class")
                        if not model_class_str:
                            raise ValueError(f"No 'class' or '_target_' field defined in 'models:{role}' cfg")
                        
                        del model_config["class"]
                        model_class_obj = self._component(model_class_str)
                        
                        # 원본 로직을 따라 모델 인스턴스 생성
                        models[agent_id][role] = model_class_obj(
                            observation_space=observation_spaces[agent_id],
                            action_space=action_spaces[agent_id],
                            device=device,
                            **self._process_cfg(model_config)) 
            else: # shared models
                raise NotImplementedError("Shared models are not implemented in this custom runner example.")
                    
            