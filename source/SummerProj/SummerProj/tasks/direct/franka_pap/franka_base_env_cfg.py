# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from dataclasses import MISSING

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.controllers.joint_impedance import JointImpedanceControllerCfg


@configclass
# class EventCfg:
#     """Configuration for randomization."""

#     # -- robot
#     robot_physics_material = EventTerm(
#         func=mdp.randomize_rigid_body_material,
#         mode="reset",
#         min_step_count_between_reset=720,
#         params={
#             "asset_cfg": SceneEntityCfg("robot"),
#             "static_friction_range": (0.7, 1.3),
#             "dynamic_friction_range": (1.0, 1.0),
#             "restitution_range": (1.0, 1.0),
#             "num_buckets": 250,
#         },
#     )
#     robot_joint_stiffness_and_damping = EventTerm(
#         func=mdp.randomize_actuator_gains,
#         min_step_count_between_reset=720,
#         mode="reset",
#         params={
#             "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
#             "stiffness_distribution_params": (0.75, 1.5),
#             "damping_distribution_params": (0.3, 3.0),
#             "operation": "scale",
#             "distribution": "log_uniform",
#         },
#     )

#     # -- object
#     object_physics_material = EventTerm(
#         func=mdp.randomize_rigid_body_material,
#         min_step_count_between_reset=720,
#         mode="reset",
#         params={
#             "asset_cfg": SceneEntityCfg("object"),
#             "static_friction_range": (0.7, 1.3),
#             "dynamic_friction_range": (1.0, 1.0),
#             "restitution_range": (1.0, 1.0),
#             "num_buckets": 250,
#         },
#     )
#     object_scale_mass = EventTerm(
#         func=mdp.randomize_rigid_body_mass,
#         min_step_count_between_reset=720,
#         mode="reset",
#         params={
#             "asset_cfg": SceneEntityCfg("object"),
#             "mass_distribution_params": (0.5, 1.5),
#             "operation": "scale",
#             "distribution": "uniform",
#         },
#     )



@configclass
class FrankaBaseEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 10.0
    decimation: int
    action_space: int
    observation_space: int
    state_space: int

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)

    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            "panda_finger_joint.*": 0.04,
        },
        ))
    # Impedance Controller를 사용하는 경우, 액추에이터 PD제어 모델 사용 X (중복 토크 계산)
    # 액추에이터에 Impedance Controller가 붙음으로써 최하단 제어기의 역할을 하게 되는 개념.
    robot.actuators["panda_shoulder"].stiffness = 0.0
    robot.actuators["panda_shoulder"].damping = 0.0
    robot.actuators["panda_forearm"].stiffness = 0.0
    robot.actuators["panda_forearm"].damping = 0.0

    # ground plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )
    
    # goal object marker
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_object",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(1.0, 1.0, 1.0),
            )
        },
    )

    goal_pos_marker_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="Visuals/goal_marker",
        markers={
        "frame": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.1, 0.1, 0.1),
            )
        }
    )

    # TCP marker
    tcp_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="Visuals/TCP_current",
        markers={
        "frame": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.1, 0.1, 0.1),
            )
        }
    )

    # Table
    table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0.0], rot=[0.707, 0, 0, 0.707]),
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # events
    # events: EventCfg = EventCfg()
    
    # Joint Impedance controller
    imp_controller: JointImpedanceControllerCfg = JointImpedanceControllerCfg(
        command_type="p_rel",
        dof_pos_offset=None,
        impedance_mode="variable",
        stiffness=300.0,
        damping_ratio=0.5,
        stiffness_limits=(0, 500),
        damping_ratio_limits=(0, 5),
        inertial_compensation=True,
        gravity_compensation=True,)
    
    # IK controller
    ik_controller = DifferentialIKControllerCfg = DifferentialIKControllerCfg(
        command_type='pose',
        use_relative_mode=True,
        ik_method='dls',)
    
    # Scene entities
    robot_entity: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=["panda_joint.*"], body_names=["panda_leftfinger"])

    # Joint Control Action scale
    loc_res_scale = 0.2
    rot_res_scale = 0.1
    joint_res_scale = 0.2
    stiffness_scale = imp_controller.stiffness_limits[1]
    damping_scale = imp_controller.damping_ratio_limits[1]

    # target point reset
    reset_position_noise_x = 0.1
    reset_position_noise_y = 0.2