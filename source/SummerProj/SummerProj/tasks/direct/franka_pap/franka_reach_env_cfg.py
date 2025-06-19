# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.markers.config import FRAME_MARKER_CFG, CUBOID_MARKER_CFG
from .franka_base_env_cfg import FrankaBaseEnvCfg


@configclass
class FrankaReachEnvCfg(FrankaBaseEnvCfg):
    # env
    episode_length_s = 10.0
    decimation = 10
    action_space = 20
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

    goal_pos_marker_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="Visuals/goal_marker",
        markers={
        "frame": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.1, 0.1, 0.1),
            )
        }
    )

    via_pos_marker_cfg: VisualizationMarkersCfg = CUBOID_MARKER_CFG.replace(
        prim_path="Visuals/via_marker")


    # reward hyperparameter
    alpha, beta = 10.0, 4.0
    w_pos = 30.0
    w_rot = 20.0
    w_penalty = 0.1