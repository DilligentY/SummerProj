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
from abc import abstractmethod

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, AssetBase
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.controllers.joint_impedance import JointImpedanceController
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate

from .franka_pap_env_cfg import FrankaPapEnvCfg

class FrankaPapBaseEnv(DirectRLEnv):
    cfg: FrankaPapEnvCfg

    def __init__(self, cfg: FrankaPapEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total env ids
        self.total_env_ids = torch.arange(self.num_envs, device=self.device)

        # Joint & Link Index
        self.joint_idx = self._robot.find_joints("panda_joint.*")[0]
        self.hand_link_idx = self._robot.find_bodies("panda_link7")[0][0]
        self.left_finger_link_idx = self._robot.find_bodies("panda_leftfinger")[0][0]
        self.left_finger_joint_idx = self._robot.find_joints("panda_finger_joint1")[0][0]
        self.right_finger_link_idx = self._robot.find_bodies("panda_rightfinger")[0][0]
        self.right_finger_joint_idx = self._robot.find_joints("panda_finger_joint2")[0][0]
        self.object_link_idx = self._object.find_bodies("Object")[0][0]
        self.finger_open_joint_pos = torch.full((self.scene.num_envs, 2), 0.04, device=self.scene.device)

        # Physics Limits
        self.num_active_joints = len(self.joint_idx)
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_dof_stiffness_lower_limits = torch.tensor(self.cfg.controller.stiffness_limits[0], device=self.device)
        self.robot_dof_stiffness_upper_limits = torch.tensor(self.cfg.controller.stiffness_limits[1], device=self.device)
        self.robot_dof_damping_lower_limits = torch.tensor(self.cfg.controller.damping_ratio_limits[0], device=self.device)
        self.robot_dof_damping_upper_limits = torch.tensor(self.cfg.controller.damping_ratio_limits[1], device=self.device)

        # Action Space
        self.robot_dof_residual = torch.zeros((self.num_envs, self.num_active_joints), device=self.device)
        self.robot_stiffness = torch.zeros((self.num_envs, self.num_active_joints), device=self.device)
        self.robot_damping_ratio = torch.zeros((self.num_envs, self.num_active_joints), device=self.device)

        # Default Object and Robot Pose
        self.robot_joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.robot_joint_vel = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.object_pos_w = torch.zeros((self.num_envs, 7), device=self.device)
        self.object_pos_b = torch.zeros((self.num_envs, 7), device=self.device)
        self.object_linvel = torch.zeros((self.num_envs, 3), device=self.device)
        self.object_angvel = torch.zeros((self.num_envs, 3), device=self.device)


        # Default TCP Pose
        self.tcp_offset = torch.tensor([0.0, 0.0, 0.045], device=self.device).repeat([self.scene.num_envs, 1])
        lfinger_pos_w = self._robot.data.body_state_w[:, self.left_finger_link_idx, :7]
        rfinger_pos_w = self._robot.data.body_state_w[:, self.right_finger_link_idx, :7]
        self.robot_grasp_pos_w = calculate_robot_tcp(lfinger_pos_w, rfinger_pos_w, self.tcp_offset)

        # Default goal positions
        self.goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_rot[:, 0] = 1.0
        self.goal_loc = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_loc[:, :] = torch.tensor([0.54, -0.64, 0.0], device=self.device)
        
        # Joint Impedance Controller
        self.controller = JointImpedanceController(cfg=self.cfg.controller,
                                                   num_robots=self.num_envs,
                                                   dof_pos_limits=self._robot.data.soft_joint_pos_limits[:, 0:self.num_active_joints, :],
                                                   device=self.device)
        
        # Goal marker
        # self.goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)

        # Tcp marker
        self.tcp_marker = VisualizationMarkers(self.cfg.tcp_cfg)

        # unit tensors
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._object = RigidObject(self.cfg.object)

        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["object"] = self._object

        spawn_ground_plane(prim_path=self.cfg.plane.prim_path, cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))
        spawn = self.cfg.table.spawn
        spawn.func(self.cfg.table.prim_path, spawn, translation=(0.5, 0.0, 0.0), orientation=(0.707, 0.0, 0.0, 0.707))
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        # robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_pos[env_ids, 0:self.num_active_joints] += sample_uniform(-0.125, 0.125,
                                                                       (len(env_ids), self.num_active_joints),
                                                                       self.device)
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # Goal State
        self._reset_target_pose(env_ids)


    # -- Oxuilary Functions
    def _reset_target_pose(self, env_ids):
        # reset goal rotation
        rand_floats = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)
        new_rot = randomize_rotation(
            rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])
        # update goal pose and markers
        self.goal_rot[env_ids] = new_rot
        goal_loc = self.goal_loc + self.scene.env_origins
        # self.goal_markers.visualize(goal_loc, self.goal_rot)


    @abstractmethod
    def _pre_physics_step(self, actions):
        raise NotImplementedError(f"Please implement the '_pre_physics_step' method for {self.__class__.__name__}.")

    @abstractmethod
    def _apply_action(self):
        raise NotImplementedError(f"Please implement the '_apply_action' method for {self.__class__.__name__}.")

    @abstractmethod
    def _get_dones(self):
        raise NotImplementedError(f"Please implement the '_get_done' method for {self.__class__.__name__}.")

    @abstractmethod
    def _get_rewards(self):
        raise NotImplementedError(f"Please implement the '_get_rewards' method for {self.__class__.__name__}.")

    @abstractmethod
    def _get_observations(self):
        raise NotImplementedError(f"Please implement the '_get_observation' method for {self.__class__.__name__}.")


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