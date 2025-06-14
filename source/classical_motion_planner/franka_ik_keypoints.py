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
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms, quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, quat_apply

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
    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 1.05], rot=[0.707, 0, 0, 0.707]),
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
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
    
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
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
        init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.5, 0.2, 1.05), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )
    



def run_simulator(sim : sim_utils.SimulationContext, scene : InteractiveScene):
    # Setup the scene
    point_marker_cfg = CUBOID_MARKER_CFG.copy()
    point_marker_cfg.markers["cuboid"].size = (0.01, 0.01, 0.01)
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

    tcp_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/tcp_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))
    traj_marker = VisualizationMarkers(point_marker_cfg.replace(prim_path="Visuals/ee_traj"))
    keypoint_marker = VisualizationMarkers(point_marker_cfg.replace(prim_path="/Visuals/keypoints"))

    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=scene.device)

    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_leftfinger"])
    robot_entity_cfg.resolve(scene)

    robot = scene["robot"]
    object = scene["object"]

    hand_link_idx = robot.find_bodies("panda_link7")[0][0]
    left_finger_link_idx = robot.find_bodies("panda_leftfinger")[0][0]
    left_finger_joint_idx = robot.find_joints("panda_finger_joint1")[0][0]
    right_finger_link_idx = robot.find_bodies("panda_rightfinger")[0][0]
    right_finger_joint_idx = robot.find_joints("panda_finger_joint2")[0][0]
    finger_open_joint_pos = torch.full((scene.num_envs, 2), 0.04, device=scene.device)
    finger_ids = torch.tensor([left_finger_joint_idx, right_finger_joint_idx], device=scene.device)


    if robot.is_fixed_base:
        jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        jacobi_idx = robot_entity_cfg.body_ids[0]

    i = 0
    sim_dt = sim.get_physics_dt()
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()
    joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()

    # Reference Frame : Robot Base Frame
    root_start_w = robot.data.root_state_w[:, :7]
    lfinger_start_w = robot.data.body_state_w[:, left_finger_link_idx, :7]
    rfinger_start_w = robot.data.body_state_w[:, right_finger_link_idx, :7]
    tcp_start_loc_w = (lfinger_start_w[:, :3] + rfinger_start_w[:, :3]) / 2.0 + quat_apply(lfinger_start_w[:, 3:7], torch.tensor([0.0, 0.0, 0.045], device=scene.device))
    tcp_start_pose_w = torch.cat((tcp_start_loc_w, lfinger_start_w[:, 3:7]), dim=1)

    tcp_start_loc_b, tcp_start_rot_b = subtract_frame_transforms(
        root_start_w[:, :3], root_start_w[:, 3:7], tcp_start_pose_w[:, :3], tcp_start_pose_w[:, 3:7])
    tcp_start_pose_b = torch.cat((tcp_start_loc_b, tcp_start_rot_b), dim=1)

    # Target End-Effector Pose -> Keypoints from the Cube Object
    object_pose_w = object.data.root_state_w[:, :7]
    goal_keypoints_loc_w = compute_keypoints(object_pose_w, num_keypoints=8)

    target_keypoint_w = select_target_keypoint(goal_keypoints_loc_w, object_pose_w)
    target_keypoint_loc_b, target_keypoint_rot_b = subtract_frame_transforms(
        root_start_w[:, :3], root_start_w[:, 3:7], target_keypoint_w[:, :3], target_keypoint_w[:, 3:7]
    )
    target_keypoint_b = torch.cat((target_keypoint_loc_b, target_keypoint_rot_b), dim=1)

    ee_goals = torch.tensor([0.3, 0.3, 0.3, 0.707, 0, 0.707, 0], device=scene.device)
    
    # Motion Planning : RRT
    # motion_planner = RRTWrapper(start=tcp_start_pose_b.squeeze_(0), goal=target_keypoint_b.squeeze_(0), env=Env.Map3D(5, 5, 5), max_dist=0.1, num_traj_points=50)
    motion_planner = RRTWrapper(start=tcp_start_pose_b.squeeze_(0), goal=ee_goals, env=Env.Map3D(5, 5, 5), max_dist=0.1, num_traj_points=50)
    optimal_trajectory = motion_planner.plan()
    
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=scene.device)
    ik_commands[:, :3] = optimal_trajectory[0, :3]
    ik_commands[:, 3:] = optimal_trajectory[0, 3:7]

    diff_ik_controller.reset()
    diff_ik_controller.set_command(ik_commands, tcp_start_pose_b[:3], tcp_start_pose_b[3:7])
    while simulation_app.is_running():

        # Get the root & joint Pose in world frame
        root_pose_w = robot.data.root_state_w[:, :7]
        # Get the tcp pose in world frame
        lfinger_pose_w = robot.data.body_state_w[:, left_finger_link_idx, :7]
        rfinger_pose_w = robot.data.body_state_w[:, right_finger_link_idx, :7]
        tcp_loc_w = (lfinger_pose_w[:, :3] + rfinger_pose_w[:, :3]) / 2.0 + quat_apply(lfinger_pose_w[:, 3:7], torch.tensor([0.0, 0.0, 0.045], device=scene.device))
        tcp_pose_w = torch.cat((tcp_loc_w, lfinger_pose_w[:, 3:7]), dim=1)
        # Compute the tcp pose in the base frame
        tcp_loc_b, tcp_rot_b = subtract_frame_transforms(
                root_pose_w[:, :3], root_pose_w[:, 3:7], tcp_pose_w[:, :3], tcp_pose_w[:, 3:7])
        tcp_pose_b = torch.cat((tcp_loc_b, tcp_rot_b), dim=1)

        # Compute the position error in the base frame
        tcp_pos_err_b = tcp_pose_b[:, :3] - ik_commands[:, :3]
        if torch.norm(tcp_pos_err_b) < 5e-2:
            i = (i+1) % optimal_trajectory.shape[0]
            if i == 0:
                # Reset the Scene
                root_state = robot.data.default_root_state.clone()
                root_state[:, :3] += scene.env_origins
                robot.write_root_pose_to_sim(root_state[:, :7])
                robot.write_root_velocity_to_sim(root_state[:, 7:])
                joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
                robot.write_joint_state_to_sim(joint_pos, joint_vel)

                object_root_state = object.data.default_root_state.clone()
                object_root_state[:, :3] += scene.env_origins
                object.write_root_pose_to_sim(object_root_state[:, :7])
                object.write_root_velocity_to_sim(object_root_state[:, 7:])

                object.reset()
                robot.reset()
                scene.reset()
                print("[INFO]: Resetting robot state...")
            ik_commands[:, :] = optimal_trajectory[i, :]
            diff_ik_controller.set_command(ik_commands, tcp_pose_b[:, :3], tcp_pose_b[:, 3:7])

        # Compute Jacobian
        jacobian = robot.root_physx_view.get_jacobians()[:, jacobi_idx, :, robot_entity_cfg.joint_ids]

        # Get the root & joint Pose in world frame
        joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]

        # Compute the desired joint position using the IK and the end-effector pose from the base frame
        joint_pos_des = diff_ik_controller.compute(tcp_loc_b, tcp_rot_b, jacobian, joint_pos)
    
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        robot.set_joint_position_target(finger_open_joint_pos, joint_ids=finger_ids)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        lfinger_pose_w = robot.data.body_state_w[:, left_finger_link_idx, :7]
        rfinger_pose_w = robot.data.body_state_w[:, right_finger_link_idx, :7]
        tcp_pose_w = (lfinger_pose_w[:, :3] + rfinger_pose_w[:, :3]) / 2.0 + quat_apply(lfinger_pose_w[:, 3:7], torch.tensor([0.0, 0.0, 0.045], device=scene.device))
        tcp_pos_w = torch.cat((tcp_pose_w, lfinger_pose_w[:, 3:7]), dim=1)

        tcp_marker.visualize(tcp_pos_w[:, :3], tcp_pos_w[:, 3:7])
        traj_marker.visualize(optimal_trajectory[:, :3] + scene.env_origins + robot.data.default_root_state[:, :3])
        keypoint_marker.visualize(goal_keypoints_loc_w.squeeze_(0))
        # goal_marker.visualize(target_keypoint_w[:, :3], target_keypoint_w[:, 3:7])
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


def select_target_keypoint(points: torch.Tensor, object_pose: torch.Tensor) -> torch.Tensor:
    rand_idx = torch.randint(0, points.shape[1], (1,),  device=points.device)
    target_point_loc = points[:, rand_idx, :]
    
    return torch.cat((target_point_loc.squeeze_(0), object_pose[:, 3:7]), dim=-1)



if __name__ == "__main__":
    main()
    simulation_app.close()