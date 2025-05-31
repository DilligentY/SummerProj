# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""This script demonstrates how to spawn Franka Emika robot & IK Test for Low & Middle-Level Controller Test.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/01_assets/run_articulation.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with an articulation.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()


# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils

from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.markers.config import FRAME_MARKER_CFG, CUBOID_MARKER_CFG
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG, FRANKA_PANDA_CFG
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from RRT.RRT_wrapper import RRTWrapper
from utils import Env



@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Design the scene Implicit Actuators on the robot."""
    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    # table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 1.05))
    )
    # robot
    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot",
                                             init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.05),
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
    



def run_simulator(sim : sim_utils.SimulationContext, scene : InteractiveScene):
    # Setup the scene
    point_marker_cfg = CUBOID_MARKER_CFG.copy()
    point_marker_cfg.markers["cuboid"].size = (0.01, 0.01, 0.01)
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))
    traj_marker = VisualizationMarkers(point_marker_cfg.replace(prim_path="Visuals/ee_traj"))
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=scene.device)
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    robot_entity_cfg.resolve(scene)
    robot = scene["robot"]

    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    robot_entity_cfg.resolve(scene)

    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]

    i = 0
    sim_dt = sim.get_physics_dt()
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()
    joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()

    # Reference Frame : Robot Base Frame
    root_start_w = robot.data.root_state_w[:, :7]
    ee_start = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], :7]
    ee_start_pos_b, ee_start_rot_b = subtract_frame_transforms(
        root_start_w[:, :3], root_start_w[:, 3:7], ee_start[:, :3], ee_start[:, 3:7])
    ee_start_b = torch.cat((ee_start_pos_b, ee_start_rot_b), dim=-1).squeeze_(0)
    ee_goals = [0.3, 0.3, 0.3, 0.707, 0, 0.707, 0]
    ee_goals = torch.tensor(ee_goals, device=scene.device)
    
    # Motion Planning : RRT
    motion_planner = RRTWrapper(start=ee_start_b, goal=ee_goals, env=Env.Map3D(5, 5, 5), max_dist=0.05)
    optimal_trajectory = motion_planner.plan()
    
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=scene.device)
    ik_commands[:, :3] = optimal_trajectory[0, :3]
    ik_commands[:, 3:] = optimal_trajectory[0, 3:7]

    diff_ik_controller.reset()
    diff_ik_controller.set_command(ik_commands, ee_start[:, :3], ee_start[:, 3:7])
    while simulation_app.is_running():
        # Get the root & joint Pose in world frame
        root_pose_w = robot.data.root_state_w[:, :7]
        # Get the end-effector pose in world frame
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], :7]
        # Compute the end-effector pose in the base frame
        ee_pos_b, ee_rot_b = subtract_frame_transforms(
            root_pose_w[:, :3], root_pose_w[:, 3:7], ee_pose_w[:, :3], ee_pose_w[:, 3:7]
        )

        # Compute the position error in the base frame
        ee_pos_err_b = ee_pos_b[:, :3] - ik_commands[:, :3]
        if torch.norm(ee_pos_err_b) < 1e-2:
            i = (i+1) % optimal_trajectory.shape[0]
            ik_commands[:, :] = optimal_trajectory[i, :]
            diff_ik_controller.set_command(ik_commands, ee_pose_w[:, :3], ee_pose_w[:, 3:7])

        # Compute Jacobian
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]

        # Get the root & joint Pose in world frame
        joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]

        # Compute the desired joint position using the IK and the end-effector pose from the base frame
        joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_rot_b, jacobian, joint_pos)
    
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], :7]
        ee_marker.visualize(ee_pose_w[:, :3], ee_pose_w[:, 3:7])
        traj_marker.visualize(optimal_trajectory[:, :3] + scene.env_origins + robot.data.default_root_state[:, :3])
        goal_marker.visualize(ee_goals[:3] + scene.env_origins + robot.data.default_root_state[:, :3], ee_goals[3:7].unsqueeze_(0))


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim_dt = sim.get_physics_dt()
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = RobotSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)

if __name__ == "__main__":
    main()
    simulation_app.close()