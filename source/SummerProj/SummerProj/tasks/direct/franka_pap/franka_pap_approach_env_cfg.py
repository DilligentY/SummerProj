# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import  RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from .franka_base_env_cfg import FrankaBaseEnvCfg


@configclass
class FrankaPapApproachEnvCfg(FrankaBaseEnvCfg):
    # env
    episode_length_s = 10.0
    decimation = 5
    action_space = 7
    observation_space = 28
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # object
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.35, 0.0, 0.0], rot=[1, 0, 0, 0]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

    # keypoints marker
    # keypoints_cfg: VisualizationMarkersCfg = CUBOID_MARKER_CFG.replace(
    #     prim_path="/Visuals/keypoint",
    #     markers={
    #     "cuboid": sim_utils.CuboidCfg(
    #         size=(0.01, 0.01, 0.01),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
    #         ),
    #     }    
    # )

    # # target keypoints marker
    # target_points_cfg: VisualizationMarkersCfg = CUBOID_MARKER_CFG.replace(
    #     prim_path="/Visuals/keypoint",
    #     markers={
    #     "cuboid": sim_utils.CuboidCfg(
    #         size=(0.01, 0.01, 0.01),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0),
    #                                                     opacity=0.1),
    #         ),
    #     }    
    # )

    # reward hyperparameter
    alpha, beta = 10.0, 4.0
    w_pos = 15.0
    w_penalty = 0.5