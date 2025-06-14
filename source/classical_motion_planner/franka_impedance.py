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
from isaaclab_assets import FRANKA_PANDA_CFG
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.controllers.joint_impedance import JointImpedanceController, JointImpedanceControllerCfg
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
    robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot",
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
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    robot_entity_cfg.resolve(scene)
    robot = scene["robot"]
    
    joint_ids = robot.find_joints(".*")[0]
    point_marker_cfg = CUBOID_MARKER_CFG.copy()
    point_marker_cfg.markers["cuboid"].size = (0.03, 0.03, 0.03)
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))

    diff_ik_cfg = DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=scene.device)
    joint_imp_cfg = JointImpedanceControllerCfg(command_type="p_abs", 
                                                impedance_mode="variable",
                                                stiffness=0.0,
                                                damping_ratio=0.0,
                                                inertial_compensation=False, 
                                                gravity_compensation=False)
    
    joint_imp_controller = JointImpedanceController(cfg=joint_imp_cfg, 
                                                    num_robots=scene.num_envs,
                                                    dof_pos_limits=robot.data.joint_pos_limits,
                                                    device=scene.device)

    # ───────── 초기 자세 저장
    sim_dt = sim.get_physics_dt()
    q_home = robot.data.default_joint_pos.clone()                # [env,7]
    q_dot  = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(q_home, q_dot)
    robot.reset()

    # # Reference Frame : Robot Base Frame
    # root_start_w = robot.data.root_state_w[:, :7]
    # ee_start = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], :7]
    # ee_start_pos_b, ee_start_rot_b = subtract_frame_transforms(
    #     root_start_w[:, :3], root_start_w[:, 3:7], ee_start[:, :3], ee_start[:, 3:7])
    # ee_start_b = torch.cat((ee_start_pos_b, ee_start_rot_b), dim=-1).squeeze_(0)
    # ee_goals = [0.3, 0.3, 0.3, 0.707, 0, 0.707, 0]
    # ee_goals = torch.tensor(ee_goals, device=scene.device)

    # ───────── 사인파 파라미터
    t = 0.0
    amp   = torch.full_like(q_home, 0.5)                         # 0.35 rad 진폭
    phase = torch.arange(amp.shape[1], device=scene.device) * (torch.pi / 4)
    freq  = 0.1
    
    # Control Command
    n_j = robot.num_joints
    kp  = 10.0
    rho = 5.0
    pos_slice       = slice(0, n_j)
    stiffness_slice = slice(n_j, 2*n_j)
    damping_slice   = slice(2*n_j, 3*n_j)
    commands = torch.zeros(scene.num_envs, joint_imp_controller.num_actions, device=scene.device)
    joint_imp_controller.set_command(commands)

    while simulation_app.is_running():
        q_des = q_home + amp * torch.sin(2*torch.pi*freq*t + phase)
        commands[:, pos_slice]       = q_des
        commands[:, stiffness_slice] = kp
        commands[:, damping_slice]   = rho
        joint_imp_controller.set_command(commands)

        torque = joint_imp_controller.compute(
            dof_pos = robot.data.joint_pos, 
            dof_vel = robot.data.joint_vel
        ) 
    
        robot.set_joint_effort_target(torque, joint_ids=joint_ids)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        t += sim_dt

        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], :7]
        ee_marker.visualize(ee_pose_w[:, :3], ee_pose_w[:, 3:7])


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