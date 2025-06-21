from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.models.torch import Model, GaussianMixin, DeterministicMixin

# --- 정책함수 클래스 ---
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
            encoder_layers.append(nn.ReLU())
            in_features = out_features
        self.encoder = nn.Sequential(*encoder_layers)

        # Policy Head
        policy_layers = []
        in_features_policy = in_features
        for out_features in policy_features:
            policy_layers.append(nn.Linear(in_features_policy, out_features))
            policy_layers.append(nn.ReLU())
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

# --- 가치함수 클래스 ---
class FrankaValue(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 encoder_features: List[int] = [256, 128],
                 value_features: List[int] = [64],
                 clip_actions: bool = False):
        
        DeterministicMixin.__init__(self, clip_actions, device)
        Model.__init__(self, observation_space, action_space, device)
        
        # Backbone
        in_features_encoder = observation_space.shape[0]
        encoder_layers = []
        for out_features in encoder_features:
            encoder_layers.append(nn.Linear(in_features_encoder, out_features))
            encoder_layers.append(nn.ReLU())
            in_features_encoder = out_features
        self.encoder = nn.Sequential(*encoder_layers)

        # Value Head
        in_features_value = encoder_features[-1]
        value_layers = []
        for out_features in value_features:
            value_layers.append(nn.Linear(in_features_value, out_features))
            value_layers.append(nn.ReLU())
            in_features_value = out_features
        self.value_branch = nn.Sequential(*value_layers)
        self.value_head = nn.Linear(in_features_value, 1)

    def compute(self, inputs: dict, role: str = "") -> tuple[torch.Tensor, ...]:
        obs = inputs["states"]
        x = self.encoder(obs)
        v = self.value_branch(x)
        return self.value_head(v), {}