# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.controllers import DifferentialIKController
from isaaclab.utils.math import subtract_frame_transforms, quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, quat_apply

from .franka_pap_base_env import FrankaPapBaseEnv
from .franka_pap_approach_env_cfg import FrankaPapApproachEnvCfg

class FrankaPapApproachEnv(FrankaPapBaseEnv):
    """Franka Pap Approach Environment for the Franka Emika Panda robot."""
    cfg: FrankaPapApproachEnvCfg
    def __init__(self, cfg: FrankaPapBaseEnv, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Action and Observation Configurations
        self.cfg.decimation = 2

        # Controller Commands & Scene Entity
        self._robot_entity = self.cfg.robot_entity
        self._robot_entity.resolve(self.scene)
        self.commands = torch.zeros((self.num_envs, self.controller.num_actions), device=self.device)

        # Object Grasp Local Frame Pose
        object_local_grasp_pose = torch.zeros((1, 8, 7), device=self.device)
        self.object_local_grasp_pos = object_local_grasp_pose[:, :, :3].repeat(self.num_envs, 1, 1)
        self.object_local_grasp_rot = object_local_grasp_pose[:, :, 3:7].repeat(self.num_envs, 1, 1)

        # Robot and Object Grasp Poses
        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.object_grasp_rot = torch.zeros((self.num_envs, 8, 4), device=self.device)
        self.object_grasp_pos = torch.zeros((self.num_envs, 8, 3), device=self.device)
        self.target_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.target_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)

        # Keypoints
        self.gt_keypoints = torch.ones(self.num_envs, 8, 3, dtype=torch.float32, device=self.device)

        # Keypoint markers
        self.keypoints_marker = VisualizationMarkers(self.cfg.keypoints_cfg)
        
        # Taret Keypoint marker
        self.target_marker = VisualizationMarkers(self.cfg.target_points_cfg)


    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        actions.shape  =  (N, 9)
        0:7  →  DOF position targets            (rad / m)
        7    →  공통 Joint-stiffness  Kp        (N·m/rad)
        8    →  공통 Damping-ratio     ζ        (–)
        """
        # ── 1. 슬라이스 & 즉시 in-place clip ──────────────────────────
        tgt  = actions[:, 0:7].clamp_(self.robot_dof_lower_limits,
                                    self.robot_dof_upper_limits)

        kp_s = actions[:, 7].clamp_(self.robot_dof_stiffness_lower_limits[0],
                                    self.robot_dof_stiffness_upper_limits[0])   # (N,)
        z_s  = actions[:, 8].clamp_(self.robot_dof_damping_lower_limits[0],
                                    self.robot_dof_damping_upper_limits[0])     # (N,)

        # ── 2. 1 값 → 7 값으로 브로드캐스트(메모리 0 복사) ────────────
        kp   = kp_s.view(-1, 1).expand(-1, 7)          # (N,7)
        damp = z_s.view(-1, 1).expand(-1, 7)           # (N,7)

        # ── 3. 로봇 버퍼에 덮어쓰기 ───────────────────────────────────
        self.robot_dof_targets.copy_(tgt)
        self.robot_stiffness    .copy_(kp)
        self.robot_damping_ratio.copy_(damp)
    
    def _apply_action(self) -> None:
        """
        최종 커맨드 [N×21] 생성 후 Actuator API 호출.
        """
        cmd = torch.cat((self.robot_dof_targets,      # (N,7)
                        self.robot_stiffness,        # (N,7) ← 공통 Kp 복제
                        self.robot_damping_ratio),   # (N,7) ← 공통 zeta 복제
                        dim=-1)                     # (N,21)
        # ==== 커맨드 세팅 ====
        self.controller.set_command(cmd)
        torque = self.controller.compute(self.robot_joint_pos,
                                         self.robot_joint_vel,
                                         mass_matrix=None,
                                         gravity=None)
        self._robot.set_joint_effort_target(torque, joint_ids=self.joint_idx)
        
        

    
    def _get_dones(self):
        # 물체가 잘 붙어있는지 Check 떨어지면, 실패 처리
        self._compute_intermediate_values()
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated
    
    def _get_rewards(self):
        # 물체 안정성과 관련한 Reward 계산
        return torch.zeros(self.num_envs, device=self.device)
    
    def _get_observations(self):
        # Object 및 Keypoints와 Robot의 상태를 포함한 Observation vector 생성
        joint_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        tcp_pos = self.robot_grasp_pos_b
        object_pos = torch.cat((self.object_loc_b, self.object_rot_b), dim=1)

        obs = torch.cat(
            (   
                # robot joint (9)
                joint_pos_scaled,
                # end effector 6D pose w.r.t Root frame (7)
                tcp_pos,
                # object position and rotiation w.r.t Root frame (7)
                object_pos,
                # ground truth object keypoints 3D pose w.r.t the Root frame (24)
                self.object_grasp_pos_b[:, :, :3].reshape(-1, 24),
            ), dim=1
        )

        return {"policy": obs}

    
    def _reset_idx(self, env_ids: torch.Tensor):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        # robot & scene reset
        super()._reset_idx(env_ids)

        # object state
        object_default_state = self._object.data.default_root_state.clone()[env_ids]
        pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)

        object_default_state[:, 0:3] = (
            object_default_state[:, 0:3] + self.cfg.reset_position_noise * pos_noise + self.scene.env_origins[env_ids]
        )

        rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)  # noise for X and Y rotation
        object_default_state[:, 3:7] = randomize_rotation(
            rot_noise[:, 0], rot_noise[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )

        object_default_state[:, 7:] = torch.zeros_like(self._object.data.default_root_state[env_ids, 7:])
        self._object.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
        self._object.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids)

        # Need to refresh the intermediate values so that _get_observations() can use the latest values
        self._compute_intermediate_values(env_ids)

    
    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        root_pos_w = self._robot.data.root_state_w[env_ids, :7]
        lfinger_pos_w = self._robot.data.body_state_w[env_ids, self.left_finger_link_idx, :7]
        rfinger_pos_w = self._robot.data.body_state_w[env_ids, self.right_finger_link_idx, :7]
        object_loc_w = self._object.data.body_pos_w[env_ids, self.object_link_idx]
        object_rot_w = self._object.data.body_quat_w[env_ids, self.object_link_idx]
        
        # data for joint
        self.robot_joint_pos = self._robot.data.joint_pos[env_ids]
        self.robot_joint_vel = self._robot.data.joint_vel[env_ids]
        self._robot.data.GRAVITY_VEC_W

        # data for TCP (world & root Frame)
        self.robot_grasp_pos_w = calculate_robot_tcp(lfinger_pos_w, rfinger_pos_w, self.tcp_offset)
        robot_grasp_loc_b, robot_grasp_rot_b = subtract_frame_transforms(
            root_pos_w[:, :3], root_pos_w[:, 3:7], self.robot_grasp_pos_w[:, :3], self .robot_grasp_pos_w[:, 3:7])
        self.robot_grasp_pos_b = torch.cat((robot_grasp_loc_b, robot_grasp_rot_b), dim=1)

        # data for object with respect to the world & root frame
        self.object_loc_w = self._object.data.root_pos_w
        self.object_rot_w = self._object.data.root_quat_w
        self.object_loc_b, self.object_rot_b = subtract_frame_transforms(
            root_pos_w[:, :3], root_pos_w[:, 3:7], self.object_loc_w, self.object_rot_w
        )
        self.object_velocities = self._object.data.root_vel_w
        self.object_linvel = self._object.data.root_lin_vel_w
        self.object_angvel = self._object.data.root_ang_vel_w

        # Visualization Keypoints
        self.tcp_marker.visualize(self.robot_grasp_pos_w[:, :3], self.robot_grasp_pos_w[:, 3:7])
        
    
    def _compute_grasp_transforms(
        self,
        root_pos: torch.Tensor,
        object_global_grasp_rot: torch.Tensor,
        object_global_grasp_loc: torch.Tensor,
    ):

        # Object Grasp Pose -> K개
        num_keypoints = object_global_grasp_loc.shape[1]
        root_pos_exp = root_pos[:, None, :].expand(-1, num_keypoints, -1)

        local_object_loc, local_object_rot = subtract_frame_transforms(
            root_pos_exp[:, :, :3], root_pos_exp[:, :, 3:7], object_global_grasp_loc, object_global_grasp_rot
        )
        
        return torch.cat((local_object_loc, local_object_rot), dim=-1)




@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )


@torch.jit.script
def compute_keypoints(
    pose: torch.Tensor,
    num_keypoints: int = 8,
    size: tuple[float, float, float] = (2 * 0.03, 2 * 0.03, 2 * 0.03),
    out: torch.Tensor | None = None,
):
    """Computes positions of 8 corner keypoints of a cube.

    Args:
        pose: Position and orientation of the center of the cube. Shape is (N, 7)
        num_keypoints: Number of keypoints to compute. Default = 8
        size: Length of X, Y, Z dimensions of cube. Default = [0.06, 0.06, 0.06]
        out: Buffer to store keypoints. If None, a new buffer will be created.
    """
    num_envs = pose.shape[0]
    if out is None:
        out = torch.ones(num_envs, num_keypoints, 3, dtype=torch.float32, device=pose.device)
    else:
        out[:] = 1.0
    for i in range(num_keypoints):
        # which dimensions to negate
        n = [((i >> k) & 1) == 0 for k in range(3)]
        corner_loc = ([(1 if n[k] else -1) * s / 2 for k, s in enumerate(size)],)
        corner = torch.tensor(corner_loc, dtype=torch.float32, device=pose.device) * out[:, i, :]
        # express corner position in the world frame
        out[:, i, :] = pose[:, :3] + quat_apply(pose[:, 3:7], corner)

    return out

@torch.jit.script
def calculate_robot_tcp(lfinger_pose_w: torch.Tensor, 
                        rfinger_pose_w: torch.Tensor, 
                        offset: torch.Tensor) -> torch.Tensor:
    tcp_loc_w = (lfinger_pose_w[:, :3] + rfinger_pose_w[:, :3]) / 2.0 + quat_apply(lfinger_pose_w[:, 3:7], offset)

    return torch.cat((tcp_loc_w, lfinger_pose_w[:, 3:7]), dim=1)