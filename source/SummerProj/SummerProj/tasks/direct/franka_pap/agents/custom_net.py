# from typing import List
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from skrl.models.torch import Model, GaussianMixin

# # skrl의 Model 클래스를 상속받아 우리의 커스텀 Actor-Critic 모델을 정의합니다.
# class FrankaCustomActorCritic(GaussianMixin, Model):
#     """
#     Franka 로봇 제어를 위한 커스텀 Actor-Critic 모델.
#     - IK 액션(6D): tanh 활성화 함수 사용
#     - 임피던스 게인(14D): softplus 활성화 함수 사용
#     - 정책(Policy)과 가치(Value) 네트워크가 초기 인코더 레이어를 공유합니다.
#     """
#     def __init__(self,
#                  observation_space,
#                  action_space,
#                  device,
#                  # Hydra 설정 파일에서 전달받을 파라미터들
#                  encoder_features: List[int] = [256, 128],
#                  policy_features: List[int] = [64],
#                  value_features: List[int] = [64],
#                  # GaussianMixin이 필요로 하는 파라미터들
#                  clip_actions: bool = False,
#                  clip_log_std: bool = True,
#                  min_log_std: float = -20.0,
#                  max_log_std: float = 2.0,
#                  reduction: str = "sum"):

#         # --- [수정 3] ---
#         # 각 부모 클래스의 __init__을 순서대로 호출합니다.
#         Model.__init__(self, observation_space, action_space, device)
#         GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
#         # --------------------

#         # --- 1. 공유 인코더 (Backbone) ---
#         # 정책망과 가치망이 공통으로 사용하는 초기 레이어들입니다.
#         in_features = observation_space.shape[0]
#         encoder_layers = []
#         for out_features in encoder_features:
#             encoder_layers.append(nn.Linear(in_features, out_features))
#             encoder_layers.append(nn.ReLU())
#             in_features = out_features
#         self.encoder = nn.Sequential(*encoder_layers)

#         # --- 2. 정책망 헤드 (Policy Head) ---
#         # 액션을 결정하는 부분입니다.
#         policy_layers = []
#         in_features_policy = in_features
#         for out_features in policy_features:
#             policy_layers.append(nn.Linear(in_features_policy, out_features))
#             policy_layers.append(nn.ReLU())
#             in_features_policy = out_features
#         self.policy_branch = nn.Sequential(*policy_layers)
        
#         # 여기서 액션의 각 부분을 위한 별도의 최종 레이어(헤드)를 정의합니다.
#         self.ik_head = nn.Linear(in_features_policy, 6)
#         self.stiffness_head = nn.Linear(in_features_policy, 7)
#         self.damping_head = nn.Linear(in_features_policy, 7)

#         # --- 3. 가치망 헤드 (Value Head) ---
#         # 상태의 가치를 평가하는 부분입니다.
#         value_layers = []
#         in_features_value = in_features
#         for out_features in value_features:
#             value_layers.append(nn.Linear(in_features_value, out_features))
#             value_layers.append(nn.ELU())
#             in_features_value = out_features
#         self.value_branch = nn.Sequential(*value_layers)
#         self.value_head = nn.Linear(in_features_value, 1)

#         # --- [수정 4] ---
#         # GaussianMixin은 log_std 파라미터를 기대합니다.
#         # 상태에 무관한(state-independent) 학습 가능한 파라미터로 정의합니다.
#         self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))
#         # --------------------

#     def compute(self, inputs: dict, role: str = "") -> tuple[torch.Tensor, ...]:
#         """
#         skrl 에이전트가 호출하는 핵심 메소드.
#         'role'에 따라 정책 또는 가치를 계산하여 반환합니다.
#         """
#         # 입력에서 상태(observation)를 가져옵니다.
#         obs = inputs["states"]
        
#         # 1. 공유 인코더 통과
#         x = self.encoder(obs)

#         # 2. 역할(role)에 따라 분기하여 계산
#         if role == "policy":
#             # 정책 브랜치를 통과
#             p = self.policy_branch(x)
            
#             # 각 헤드에 맞는 활성화 함수 적용 [tanh, softplus, softplus]
#             ik_actions = torch.tanh(self.ik_head(p))
#             stiffness_actions = F.softplus(self.stiffness_head(p))
#             damping_actions = F.softplus(self.damping_head(p))
            
#             # 액션들을 하나로 결합하여 최종 액션 벡터 생성
#             mean_actions = torch.cat([ik_actions, stiffness_actions, damping_actions], dim=-1)
            
#             # 이제 GaussianMixin의 요구사항에 맞춰 mean과 log_std를 모두 반환합니다.
#             return mean_actions, self.log_std_parameter, {} 
        
#         elif role == "value":
#             # 가치 브랜치를 통과하여 가치(value) 추정
#             v = self.value_branch(x)
#             value = self.value_head(v)
#             return value, None, {}

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.models.torch import Model, GaussianMixin, DeterministicMixin

# --- [수정 1: 정책(Actor)을 위한 클래스] ---
class FrankaGaussianPolicy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 encoder_features: List[int] = [256, 128],
                 policy_features: List[int] = [64],
                 clip_actions: bool = False,
                 clip_log_std: bool = True,
                 min_log_std: float = -20.0,
                 max_log_std: float = 2.0):
        
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        # Backbone
        in_features = observation_space.shape[0]
        encoder_layers = []
        for out_features in encoder_features:
            encoder_layers.append(nn.Linear(in_features, out_features))
            encoder_layers.append(nn.ELU())
            in_features = out_features
        self.encoder = nn.Sequential(*encoder_layers)

        # 정책망 헤드 (Policy Head)
        policy_layers = []
        in_features_policy = in_features
        for out_features in policy_features:
            policy_layers.append(nn.Linear(in_features_policy, out_features))
            policy_layers.append(nn.ELU())
            in_features_policy = out_features
        self.policy_branch = nn.Sequential(*policy_layers)
        
        self.ik_head = nn.Linear(in_features_policy, 6)
        self.stiffness_head = nn.Linear(in_features_policy, 7)
        self.damping_head = nn.Linear(in_features_policy, 7)

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs: dict, role: str = "") -> tuple[torch.Tensor, ...]:
        obs = inputs["states"]
        x = self.encoder(obs)
        p = self.policy_branch(x)
        
        ik_actions = torch.tanh(self.ik_head(p))
        stiffness_actions = F.tanh(self.stiffness_head(p))
        damping_actions = F.tanh(self.damping_head(p))
        mean_actions = torch.cat([ik_actions, stiffness_actions, damping_actions], dim=-1)
        
        return mean_actions, self.log_std_parameter, {}

# --- [수정 2: 가치(Critic)를 위한 클래스] ---
class FrankaValue(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 encoder_features: List[int] = [256, 128],
                 value_features: List[int] = [64],
                 clip_actions: bool = False):
        
        DeterministicMixin.__init__(self, clip_actions, device)
        Model.__init__(self, observation_space, action_space, device)
        
        # --- [수정 2] ---
        # 정책망과 동일한 구조의 인코더를 스스로 생성합니다.
        # 이제 이 인코더는 정책망의 인코더와는 별개의 객체입니다.
        in_features_encoder = observation_space.shape[0]
        encoder_layers = []
        for out_features in encoder_features:
            encoder_layers.append(nn.Linear(in_features_encoder, out_features))
            encoder_layers.append(nn.ELU())
            in_features_encoder = out_features
        self.encoder = nn.Sequential(*encoder_layers)
        # --------------------

        in_features_value = encoder_features[-1]
        value_layers = []
        for out_features in value_features:
            value_layers.append(nn.Linear(in_features_value, out_features))
            value_layers.append(nn.ELU())
            in_features_value = out_features
        self.value_branch = nn.Sequential(*value_layers)
        self.value_head = nn.Linear(in_features_value, 1)

    def compute(self, inputs: dict, role: str = "") -> tuple[torch.Tensor, ...]:
        obs = inputs["states"]
        x = self.encoder(obs)
        v = self.value_branch(x)
        return self.value_head(v), {}