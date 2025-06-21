# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch
from isaaclab.utils.math import quat_error_magnitude, subtract_frame_transforms, quat_from_angle_axis, quat_mul, sample_uniform, quat_apply, saturate
from isaaclab.markers import VisualizationMarkers

from .franka_base_env import FrankaBaseEnv
from .franka_reach_env_cfg import FrankaReachEnvCfg

class FrankaReachEnv(FrankaBaseEnv):
    """Franka Pap Approach Environment for the Franka Emika Panda robot."""
    cfg: FrankaReachEnvCfg
    def __init__(self, cfg: FrankaBaseEnv, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Controller Commands & Scene Entity
        self._robot_entity = self.cfg.robot_entity
        self._robot_entity.resolve(self.scene)
        self.processed_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.imp_commands = torch.zeros((self.num_envs, self.imp_controller.num_actions), device=self.device)
        self.ik_commands = torch.zeros((self.num_envs, self.ik_controller.action_dim), device=self.device)

        # Parameter for IK Controller
        if self._robot.is_fixed_base:
            self.jacobi_idx = self._robot_entity.body_ids[0] - 1
        else:
            self.jacobi_idx = self._robot_entity.body_ids[0]

        # Goal pose
        self.goal_pos_w = torch.zeros((self.num_envs, 7), device=self.device)
        self.goal_pos_b = torch.zeros((self.num_envs, 7), device=self.device)

        # Robot and Object Grasp Poses
        self.robot_grasp_pos_w = torch.zeros((self.num_envs, 7), device=self.device)
        self.robot_grasp_pos_b = torch.zeros((self.num_envs, 7), device=self.device)

        # Object Move Checker & Success Checker
        self.prev_loc_error = torch.zeros(self.num_envs, device=self.device)
        self.prev_rot_error = torch.zeros(self.num_envs, device=self.device)
        self.loc_error = torch.zeros(self.num_envs, device=self.device)
        self.rot_error = torch.zeros(self.num_envs, device=self.device)
        self.is_reach = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Domain Randomization Scale
        self.noise_scale = torch.tensor(
                            [self.cfg.reset_position_noise_x, self.cfg.reset_position_noise_y],
                            device=self.device,)
        
        # Goal point & Via point marker
        self.target_marker = VisualizationMarkers(self.cfg.goal_pos_marker_cfg)
        self.via_marker = VisualizationMarkers(self.cfg.via_pos_marker_cfg)


    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        actions.shape  =  (N, 20)
        0:6      →  Delta EE 6D Pose for IK                       (m, -)
        6:13     →  Joint-stiffness for Impedance Control   (N·m/rad)
        13:20    →  Damping-ratio for Impedance Control     (-)
        """
        self.actions = actions.clone()
        # ── 슬라이스 & 즉시 in-place clip ──────────────────────────
        self.processed_actions[:, :3] = self.actions[:, :3] * self.cfg.loc_res_scale
        self.processed_actions[:, 3:6] = self.actions[:, 3:6] * self.cfg.rot_res_scale
        self.processed_actions[:, 6:13] = torch.clamp(self.actions[:, 6:13] * self.cfg.stiffness_scale,
                                                      self.robot_dof_stiffness_lower_limits,
                                                      self.robot_dof_stiffness_upper_limits)
        self.processed_actions[:, 13:] = torch.clamp(self.actions[:, 13:] * self.cfg.damping_scale,
                                                     self.robot_dof_damping_lower_limits,
                                                     self.robot_dof_damping_upper_limits) 
        
        # ===== IK Command 세팅 =====
        # print(f"delta_pose = {self.processed_actions[0, :6]}")
        self.ik_controller.set_command(self.processed_actions[:, :6],
                                       self.robot_grasp_pos_b[:, :3],
                                       self.robot_grasp_pos_b[:, 3:7])
        
        # ===== Impedance Controller Gain 세팅 =====
        self.imp_commands[:,   self.num_active_joints : 2*self.num_active_joints] = self.processed_actions[:, 6:13]
        self.imp_commands[:, 2*self.num_active_joints : 3*self.num_active_joints] = self.processed_actions[:, 13:]

        # print(f"kp = {self.processed_actions[0, 6:13]}")
        # print(f"damping= {self.processed_actions[0, 13:]}")
        

    def _apply_action(self) -> None:
        """
            최종 커맨드 [N x 20] 생성 후 Controller API 호출.
        """
        # ========= Data 세팅 ==========
        robot_root_pos = self._robot.data.root_state_w[:, :7]
        robot_joint_pos = self._robot.data.joint_pos[:, :self.num_active_joints]
        robot_joint_vel = self._robot.data.joint_vel[:, :self.num_active_joints]
        lfinger_pos_w = self._robot.data.body_state_w[:, self.left_finger_link_idx, :7]
        rfinger_pos_w = self._robot.data.body_state_w[:, self.right_finger_link_idx, :7]
        robot_grasp_pos_w = calculate_robot_tcp(lfinger_pos_w, rfinger_pos_w, self.tcp_offset)
        robot_grasp_loc_b, robot_grasp_rot_b = subtract_frame_transforms(
            robot_root_pos[:, :3], robot_root_pos[:, 3:7], robot_grasp_pos_w[:, :3], robot_grasp_pos_w[:, 3:7])

        gen_mass = self._robot.root_physx_view.get_generalized_mass_matrices()[:, :self.num_active_joints, :self.num_active_joints]
        gen_grav = self._robot.root_physx_view.get_gravity_compensation_forces()[:, :self.num_active_joints]

        # ========= Inverse Kinematics =========
        jacobian = self._robot.root_physx_view.get_jacobians()[:, self.jacobi_idx, :, :self.num_active_joints]
        # Target Joint Angle 계산
        joint_pos_des = self.ik_controller.compute(robot_grasp_loc_b, 
                                                   robot_grasp_rot_b, 
                                                   jacobian, 
                                                   robot_joint_pos)
    
        # ======== Joint Impedance Regulator ========
        # 안정적 학습을 위한 Joint Cliiping (이후, Residual Curriculum으로 확장 예정)
        res_joint_pos = saturate(joint_pos_des - robot_joint_pos,
                                 self.robot_dof_res_lower_limits, 
                                 self.robot_dof_res_upper_limits)
        
        self.imp_commands[:, :self.num_active_joints] = res_joint_pos
        self.imp_controller.set_command(self.imp_commands)
        des_torque = self.imp_controller.compute(dof_pos=robot_joint_pos,
                                                 dof_vel=robot_joint_vel,
                                                 mass_matrix=gen_mass,
                                                 gravity=gen_grav)
        
        # ===== Target Torque 버퍼에 저장 =====
        self._robot.set_joint_effort_target(des_torque, joint_ids=self.joint_idx)
        
    def _get_dones(self):
        self._compute_intermediate_values()
        self.is_reach = torch.logical_and(self.loc_error < 1e-2, self.rot_error < 1e-2)
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        terminated = self.is_reach
        return terminated, truncated
        
    def _get_rewards(self):
        # Action Penalty
        action_norm = torch.norm(self.actions[:, 6:13], dim=1)

        # =========== Approach Reward (1): Potential Based Reward Shaping =============
        gamma = 0.99
        phi_s_prime = torch.exp(0.3 * self.loc_error)
        phi_s = torch.exp(0.3 * self.prev_loc_error)

        phi_s_prime_rot = torch.exp(0.2 * self.rot_error)
        phi_s_rot = torch.exp(0.2 * self.prev_rot_error)

        r_pos = gamma*phi_s_prime - phi_s 
        r_rot = gamma*phi_s_prime_rot - phi_s_rot

        # ========== Approach Reward (2): Distance Reward Shaping ===========
        # r_pos = 1 - torch.tanh(self.loc_error/3.0)
        # r_rot = 1 - torch.tanh(self.rot_error/0.5)

        # print(f"pos_error : {self.loc_error[0]}")
        # print(f"pos_reward : {r_pos[0]}")


        # =========== Success Reward : Goal Reach ============
        r_success = self.is_reach.float()
        
        # =========== Summation =============
        # IK에서 지정한 Command값을 얼마나 잘 추종했는가? 에 대한 보상도 함께 있으면 좋아보인다.. 우선 보류
        reward = self.cfg.w_pos * r_pos + self.cfg.w_rot * r_rot - self.cfg.w_penalty * action_norm + r_success

        # print(f"reward of env1 : {reward[0]}")
        # print(f"--------------------------------------")

        return reward
    
    def _get_observations(self):
        # Object 및 Robot의 상태를 포함한 Observation vector 생성
        joint_pos_scaled = (
            2.0
            * (self.robot_joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )

        # object_loc_tcp, object_rot_tcp = subtract_frame_transforms(
        #     self.robot_grasp_pos_w[:, :3], self.robot_grasp_pos_w[:, 3:7], self.goal_pos_w[:, :3], self.goal_pos_w[:, 3:7]
        # )

        # object_pos_tcp = torch.cat([object_loc_tcp, object_rot_tcp], dim=1)

        obs = torch.cat(
            (   
                # robot joint pose (7 not 9)
                joint_pos_scaled[:, 0:self.num_active_joints],
                # robot joint velocity (7 not 9)
                self.robot_joint_vel[:, 0:self.num_active_joints],
                # TCP 6D pose w.r.t Root frame (7)
                self.robot_grasp_pos_b,
                # object position w.r.t Root frame (7)
                self.goal_pos_b,
            ), dim=1
        )

        return {"policy": obs}

    
    def _reset_idx(self, env_ids: torch.Tensor):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        # ============ Robot State & Scene 리셋 ===============
        super()._reset_idx(env_ids)
        self.ik_controller.reset(env_ids)

        # ============ Target Point 리셋 ===============
        # object(=target point) reset : Location
        loc_noise_x = sample_uniform(0.5, 0.8, (len(env_ids), 1), device=self.device)
        loc_noise_y = sample_uniform(-0.3, 0.3, (len(env_ids), 1), device=self.device)
        loc_noise_z = sample_uniform(0.1, 0.5, (len(env_ids), 1), device=self.device)
        loc_noise = torch.cat([loc_noise_x, loc_noise_y, loc_noise_z], dim=-1)
        object_default_state = torch.zeros_like(self._robot.data.root_state_w[env_ids], device=self.device)
        object_default_state[:, :3] += loc_noise + self.scene.env_origins[env_ids, :3]
    
        # object(=target point) reset : Rotation
        rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        object_default_state[:, 3:7] = randomize_rotation(
            rot_noise[:, 0], rot_noise[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )

        # Convert from World to Root Frame
        root_pos_w = self._robot.data.root_state_w[env_ids, :7]
        goal_loc_b, goal_rot_b = subtract_frame_transforms(
            root_pos_w[:, :3], root_pos_w[:, 3:7], object_default_state[:, :3], object_default_state[:, 3:7])

        # Setting Target point 6D pose
        self.goal_pos_w[env_ids] = object_default_state[:, :7]
        self.goal_pos_b[env_ids] = torch.cat([goal_loc_b, goal_rot_b], dim=-1)
        
        # Need to refresh the intermediate values so that _get_observations() can use the latest values
        self._compute_intermediate_values(env_ids)

    
    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        # ========= TCP 업데이트 ===========
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

        # ========= Position Error 업데이트 =========
        # Location
        self.prev_loc_error[env_ids] = self.loc_error[env_ids]
        self.loc_error[env_ids] = torch.norm(
            self.robot_grasp_pos_b[env_ids, :3] - self.goal_pos_b[env_ids, :3], dim=1)
        # print(f"Distnace : {self.loc_error[0]}")
        # Rotation
        self.prev_rot_error[env_ids] = self.rot_error[env_ids]
        self.rot_error[env_ids] = quat_error_magnitude(self.robot_grasp_pos_b[env_ids, 3:7], self.goal_pos_b[env_ids, 3:7])
        
        # ======== Visualization ==========
        self.via_marker.visualize(self.processed_actions[:, :3] + self.robot_grasp_pos_w[:, :3])
        self.tcp_marker.visualize(self.robot_grasp_pos_w[:, :3], self.robot_grasp_pos_w[:, 3:7])
        self.target_marker.visualize(self.goal_pos_b[:, :3] + root_pos_w[:, :3], self.goal_pos_w[:, 3:7])
        

@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )

@torch.jit.script
def calculate_robot_tcp(lfinger_pose_w: torch.Tensor, 
                        rfinger_pose_w: torch.Tensor, 
                        offset: torch.Tensor) -> torch.Tensor:
    tcp_loc_w = (lfinger_pose_w[:, :3] + rfinger_pose_w[:, :3]) / 2.0 + quat_apply(lfinger_pose_w[:, 3:7], offset)

    return torch.cat((tcp_loc_w, lfinger_pose_w[:, 3:7]), dim=1)