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
from isaaclab.utils.math import quat_error_magnitude, subtract_frame_transforms, quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, quat_apply

from .franka_pap_base_env import FrankaPapBaseEnv
from .franka_pap_approach_env_cfg import FrankaPapApproachEnvCfg

class FrankaPapApproachEnv(FrankaPapBaseEnv):
    """Franka Pap Approach Environment for the Franka Emika Panda robot."""
    cfg: FrankaPapApproachEnvCfg
    def __init__(self, cfg: FrankaPapBaseEnv, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Controller Commands & Scene Entity
        self._robot_entity = self.cfg.robot_entity
        self._robot_entity.resolve(self.scene)
        self.commands = torch.zeros((self.num_envs, self.controller.num_actions), device=self.device)
        self.ik_commands = torch.zeros((self.num_envs, self.ik_controller.action_dim), device=self.device)

        # Parameter for IK Controller
        if self._robot.is_fixed_base:
            self.jacobi_idx = self._robot_entity.body_ids[0] - 1
        else:
            self.jacobi_idx = self._robot_entity.body_ids[0]

        # Object Grasp Local Frame Pose
        object_local_grasp_pose = torch.zeros((1, 8, 7), device=self.device)
        self.object_local_grasp_pos = object_local_grasp_pose[:, :, :3].repeat(self.num_envs, 1, 1)
        self.object_local_grasp_rot = object_local_grasp_pose[:, :, 3:7].repeat(self.num_envs, 1, 1)

        # Robot and Object Grasp Poses
        self.robot_grasp_pos_w = torch.zeros((self.num_envs, 7), device=self.device)
        self.robot_grasp_pos_b = torch.zeros((self.num_envs, 7), device=self.device)
        self.target_grasp_pos_w = torch.zeros((self.num_envs, 4), device=self.device)
        self.target_grasp_pos_b = torch.zeros((self.num_envs, 3), device=self.device)

        # Object Move Checker
        self.loc_error = torch.zeros(self.num_envs, device=self.device)
        self.rot_error = torch.zeros(self.num_envs, device=self.device)
        self.is_object_move = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.noise_scale = torch.tensor(
                            [self.cfg.reset_position_noise_x, self.cfg.reset_position_noise_y],
                            device=self.device,)

        # # Keypoints
        # self.gt_keypoints = torch.ones(self.num_envs, 8, 3, dtype=torch.float32, device=self.device)

        # # Keypoint markers
        # self.keypoints_marker = VisualizationMarkers(self.cfg.keypoints_cfg)
        
        # # Taret Keypoint marker
        # self.target_marker = VisualizationMarkers(self.cfg.target_points_cfg)


    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        actions.shape  =  (N, 9)
        0 :6     →  DOF position targets       (rad / m)
        7 :14    →  Joint-stiffness  Kp        (N·m/rad)
        14:21    →  Damping-ratio     ζ        (-)
        """
        self.actions = actions.clone().clamp(-1.0, 1.0)
        # ── 1. 슬라이스 & 즉시 in-place clip ──────────────────────────
        self.ik_commands[:, :3] = self.actions[:, :3] * self.cfg.loc_res_scale
        self.ik_commands[:, 3:] = self.actions[:, 3:] * self.cfg.rot_res_scale
        self.ik_controller.set_command(self.ik_commands, self.robot_grasp_pos_b[:, :3], self.robot_grasp_pos_b[:, 3:7])

        # kp_s = self.cfg.stiffness_scale * self.actions[:, 6:13]
        # kp_s = kp_s.clamp(self.robot_dof_stiffness_lower_limits,
        #                   self.robot_dof_stiffness_upper_limits)

        # z_s  = self.cfg.damping_scale * self.actions[:, 13:]
        # z_s  = z_s.clamp(self.robot_dof_damping_lower_limits,
        #                  self.robot_dof_damping_upper_limits)

        # ── 2. 1 값 → 7 값으로 브로드캐스트 ────────────
        # kp   = kp_s.view(-1, 1).expand(-1, self.num_active_joints)          # (N,7)
        # damp = z_s.view(-1, 1).expand(-1, self.num_active_joints)           # (N,7)

        # ── 3. 로봇 버퍼에 덮어쓰기 ───────────────────────────────────
        # self.robot_dof_residual.copy_(tgt)
        # self.robot_stiffness    .copy_(kp_s)
        # self.robot_damping_ratio.copy_(z_s)

        # print(f"delta joint : {self.robot_dof_residual[0, :]}")

    def _apply_action(self) -> None:
        """
        최종 커맨드 [N x 21] 생성 후 Actuator API 호출.
        """
        # ====== 유효 조인트에 대한 자코비안 계산 ======
        jacobian = self._robot.root_physx_view.get_jacobians()[:, self.jacobi_idx, :, :self.num_active_joints]

        # Desired Joint 각도 계산 with respect to root frame
        joint_pos_des = self.ik_controller.compute(self.robot_grasp_pos_b[:, :3], 
                                                   self.robot_grasp_pos_b[:, 3:7], 
                                                   jacobian, 
                                                   self.robot_joint_pos[:, :self.num_active_joints])
        
        # Desried Joint 각도 발행 
        self._robot.set_joint_position_target(joint_pos_des, joint_ids=self.joint_idx[:self.num_active_joints])


        # # ==== 커맨드 세팅 ====
        # # -- 1) Residual 값의 급격한 변화를 방지하기 위한 LPF --
        # filtered_residual = row_pass_filter(self.robot_dof_residual,
        #                                     self.robot_prev_dof_residual,
        #                                     self.physics_dt,
        #                                     omega=torch.tensor(50.0, device=self.device))
        
        # cmd = torch.cat((filtered_residual,      # (N,7)
        #                 self.robot_stiffness,        # (N,7) 
        #                 self.robot_damping_ratio),   # (N,7)
        #                 dim=-1)                     # (N,21)

        # # -- 2) LPF 기반으로 계산된 Joint Residual을 command로 입력 --
        # self.controller.set_command(cmd)
        # torque = self.controller.compute(self.robot_joint_pos[:, 0:self.num_active_joints],
        #                                  self.robot_joint_vel[:, 0:self.num_active_joints],
        #                                  mass_matrix=None,
        #                                  gravity=None)

        # self._robot.set_joint_effort_target(torque, joint_ids=self.joint_idx)

        # # -- 3) Recursive Filter의 특성을 반영, prev값 업데이트
        # self.robot_prev_dof_residual = filtered_residual
        
    
    def _get_dones(self):
        self._compute_intermediate_values()
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        terminated = self.is_object_move
        return terminated, truncated
        
    def _get_rewards(self):
        # Action Penalty
        # kp_norm = self.actions[:, 7]                   # [-1,1] → Kp 비선형 스케일 전 값
        # kp_pen      = kp_norm ** 2                     # (env,)
        joint_vel_norm = torch.norm(self.robot_joint_vel[:, :self.num_active_joints], dim=1)
        # Object Contact Penalty
        penalty_move = self.is_object_move.float()
        # Approach Reward : Distance Nomarlization
        r_pos = 1 - torch.tanh(self.loc_error/0.2)
        # Success Reward : Goal Reach
        if self.loc_error < 1e-2 and self.rot_error < 1e-3:
            r_success = torch.tensor(1.0, device=self.device) 
        else:
            r_success = torch.tensor(0.0, device=self.device)
        
        reward = self.cfg.w_pos * r_pos - self.cfg.w_penalty * joint_vel_norm - penalty_move + r_success

        # print(f"reward of env1 : {reward[0]}")

        return reward
    
    def _get_observations(self):
        # Object 및 Robot의 상태를 포함한 Observation vector 생성
        joint_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )

        object_loc_tcp, object_rot_tcp = subtract_frame_transforms(
            self.robot_grasp_pos_w[:, :3], self.robot_grasp_pos_w[:, 3:7], self.object_pos_w[:, :3], self.object_pos_w[:, 3:7]
        )

        object_pos_tcp = torch.cat([object_loc_tcp, object_rot_tcp], dim=1)

        obs = torch.cat(
            (   
                # robot joint pose (7 not 9)
                joint_pos_scaled[:, 0:self.num_active_joints],
                # robot joint velocity (7 not 9)
                self.robot_joint_vel[:, 0:self.num_active_joints],
                # TCP 6D pose w.r.t Root frame (7)
                self.robot_grasp_pos_b,
                # object position and rotiation w.r.t TCP Frame (7)
                object_pos_tcp,
            ), dim=1
        )

        return {"policy": obs}

    
    def _reset_idx(self, env_ids: torch.Tensor):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        # robot & scene reset
        super()._reset_idx(env_ids)
        self.ik_controller.reset(env_ids)

        # object state reset : X and Y Location
        # X Range : 0.2 + [-0.1, 0.1] = [0.1, 0.3]
        # y Range : 0.0 + [-0.2, 0.2] = [-0.2, 0.2]
        object_default_state = self._object.data.default_root_state.clone()[env_ids]

        loc_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)
        object_default_state[:, :2] += self.noise_scale * loc_noise + self.scene.env_origins[env_ids, :2]
    
        # object state reset : X and Y Rotation
        rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)
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
        
        # data for joint
        self.robot_joint_pos[env_ids] = self._robot.data.joint_pos[env_ids]
        self.robot_joint_vel[env_ids] = self._robot.data.joint_vel[env_ids]

        # data for TCP (world & root Frame)
        self.robot_grasp_pos_w[env_ids] = calculate_robot_tcp(lfinger_pos_w, rfinger_pos_w, self.tcp_offset[env_ids])
        robot_grasp_loc_b, robot_grasp_rot_b = subtract_frame_transforms(
            root_pos_w[:, :3], root_pos_w[:, 3:7], self.robot_grasp_pos_w[env_ids, :3], self.robot_grasp_pos_w[env_ids, 3:7])
        self.robot_grasp_pos_b[env_ids] = torch.cat((robot_grasp_loc_b, robot_grasp_rot_b), dim=1)

        # data for object with respect to the world & root frame
        object_loc_w = self._object.data.root_pos_w[env_ids]
        object_rot_w = self._object.data.root_quat_w[env_ids]
        object_loc_b, object_rot_b = subtract_frame_transforms(
            root_pos_w[:, :3], root_pos_w[:, 3:7], object_loc_w, object_rot_w
        )
        self.object_pos_w[env_ids] = torch.cat([object_loc_w, object_rot_w], dim=-1)
        self.object_pos_b[env_ids] = torch.cat([object_loc_b, object_rot_b], dim=-1)
        self.object_linvel[env_ids] = self._object.data.root_lin_vel_w[env_ids]
        self.object_angvel[env_ids] = self._object.data.root_ang_vel_w[env_ids]

        # Whether Contact
        self.loc_error[env_ids] = torch.norm(
            self.robot_grasp_pos_b[env_ids, :3] - object_loc_b[:, :3], dim=1
        )
        self.rot_error[env_ids] = quat_error_magnitude(self.robot_grasp_pos_b[env_ids, 3:7], object_rot_b[:, 3:7])
        self.is_object_move[env_ids] = torch.logical_and(self.loc_error[env_ids] < 1e-1,
                                                        torch.logical_or(torch.norm(self.object_angvel[env_ids], dim=1) > 1e-3, 
                                                                         torch.norm(self.object_linvel[env_ids], dim=1) > 1e-3))

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

@torch.jit.script
def row_pass_filter(target_residual: torch.Tensor,
                    prev_residual: torch.Tensor,
                    time_step: float,
                    omega: torch.Tensor) -> torch.Tensor:
    
    return 1 / (1+omega * time_step) * (prev_residual + omega * time_step * target_residual)