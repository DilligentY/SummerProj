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
from isaaclab.controllers.joint_impedance import JointImpedanceController, JointImpedanceControllerCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms, quat_conjugate, quat_from_angle_axis, quat_apply, quat_error_magnitude

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
        )
        )
    # Impedance Controller를 사용하는 경우, 액추에이터 PD제어 모델 사용 X (중복 토크 계산)
    # 액추에이터에 Impedance Controller가 붙음으로써 최하단 제어기의 역할을 하게 되는 개념.
    robot.actuators["panda_shoulder"].stiffness = 0.0
    robot.actuators["panda_shoulder"].damping = 0.0
    robot.actuators["panda_forearm"].stiffness = 0.0
    robot.actuators["panda_forearm"].damping = 0.0
    
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
    # ============== Setup the scene =================
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_leftfinger"])
    robot_entity_cfg.resolve(scene)

    robot = scene["robot"]
    object = scene["object"]

    point_marker_cfg = CUBOID_MARKER_CFG.copy()
    point_marker_cfg.markers["cuboid"].size = (0.01, 0.01, 0.01)
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

    tcp_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/tcp_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))
    traj_marker = VisualizationMarkers(point_marker_cfg.replace(prim_path="Visuals/ee_traj"))

    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=scene.device)

    num_active_joint = len(robot_entity_cfg.joint_ids)
    joint_imp_cfg = JointImpedanceControllerCfg(command_type="p_rel", 
                                                impedance_mode="variable",
                                                stiffness=400.0,
                                                damping_ratio=1.0,
                                                inertial_compensation=True, 
                                                gravity_compensation=True)
    joint_imp_controller = JointImpedanceController(cfg=joint_imp_cfg, 
                                                    num_robots=scene.num_envs,
                                                    dof_pos_limits=robot.data.joint_pos_limits[:, :num_active_joint],
                                                    device=scene.device)
    
    # =============== Link and Joint Indexs =================
    joint_ids = robot_entity_cfg.joint_ids
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

    
    # ================ Initial Setting ====================
    # ---------- 환경 준비 ----------
    sim_dt = scene.physics_dt
    n_j          = 7           # Franka: 9 (팔7+그리퍼2) 또는 7
    kp_table = torch.tensor([100, 80, 80, 80, 80, 80, 80], device=scene.device)
    zeta         = 0.3                       # Damping ratio(=가상 댐퍼 비율)
    joint_limits = robot.data.joint_pos_limits
    offset = torch.tensor([0.0, 0.0, 0.045], device=scene.device).unsqueeze(0)

    # ---------- 슬라이스 정의 ----------
    pos_slice       = slice(0, n_j)
    stiff_slice     = slice(n_j, 2*n_j)
    damp_slice      = slice(2*n_j, 3*n_j)

    # ---------- 명령 버퍼 ----------
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=scene.device)
    imp_commands = torch.zeros(scene.num_envs,
                               joint_imp_controller.num_actions,
                               device=scene.device)
    
    # ---------- 초기값 설정 ----------
    zero_joint_efforts = torch.zeros(scene.num_envs, n_j, device=sim.device)
    q_init   = robot.data.default_joint_pos.clone()
    q_dot_init = robot.data.default_joint_vel.clone()
    q_target = q_init[:, :n_j].clone()

    # --------- 로봇 및 오브젝트 초기화 ---------
    joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_effort_target(zero_joint_efforts, joint_ids=joint_ids)
    robot.write_data_to_sim()

    object_root_state = object.data.default_root_state.clone()
    object_root_state[:, :3] += scene.env_origins
    object.write_root_pose_to_sim(object_root_state[:, :7])
    object.write_root_velocity_to_sim(object_root_state[:, 7:])

    object.reset()
    robot.reset()
    scene.reset()
    robot.update(sim_dt)

    # ============ Motion Planner Setting (RRT & IK) ===============
    # ------- TCP 6D Pose 정의 (월드 프레임, Root 프레임) -------
    root_start_w = robot.data.root_state_w[:, :7]
    lfinger_start_w = robot.data.body_state_w[:, left_finger_link_idx, :7]
    rfinger_start_w = robot.data.body_state_w[:, right_finger_link_idx, :7]
    tcp_start_loc_w = (lfinger_start_w[:, :3] + rfinger_start_w[:, :3]) / 2.0 + quat_apply(lfinger_start_w[:, 3:7], torch.tensor([0.0, 0.0, 0.045], device=scene.device))
    tcp_start_pose_w = torch.cat((tcp_start_loc_w, lfinger_start_w[:, 3:7]), dim=1)
    tcp_start_loc_b, tcp_start_rot_b = subtract_frame_transforms(
        root_start_w[:, :3], root_start_w[:, 3:7], tcp_start_pose_w[:, :3], tcp_start_pose_w[:, 3:7])
    tcp_start_pose_b = torch.cat((tcp_start_loc_b, tcp_start_rot_b), dim=1)

    # ------- Target End-Effector Position 정의 --------
    object_pose_w = object.data.root_state_w[:, :7]
    object_loc_b, object_rot_b = subtract_frame_transforms(
         root_start_w[:, :3], root_start_w[:, 3:7], object_pose_w[:, :3], object_pose_w[:, 3:7])
    object_pose_b = torch.cat([object_loc_b, object_rot_b], dim=1)
    object_pose_b[:, 1] -= 0.6
    object_pose_b[:, 2] += 0.3
    object_pose_b[:, 3:7] = torch.tensor([0.3, 0.2, 0.0, 0.0], device=scene.device)
    
    # ------- Trajectory Planner : RRT -------
    motion_planner = RRTWrapper(start=tcp_start_pose_b.squeeze_(0), goal=object_pose_b.squeeze_(0), env=Env.Map3D(5, 5, 5), max_dist=0.1, num_traj_points=50)
    optimal_trajectory = motion_planner.plan()

    # ------- Motion Planner : Inverse Kinematics -------
    ik_commands[:, :3] = optimal_trajectory[0, :3]
    ik_commands[:, 3:] = optimal_trajectory[0, 3:7]
    diff_ik_controller.reset()
    diff_ik_controller.set_command(ik_commands, tcp_start_pose_b[:3], tcp_start_pose_b[3:7])



    # ========== Low-Level Controller (Joint Impadance Regulation) ===========
    # ------- Impedance Control Command 세팅 -------
    imp_commands[:, 7:14] = kp_table
    imp_commands[:, 14:] = zeta



    # ========== 제어 루프 ======================================
    i = 0
    while simulation_app.is_running():
        
        # --------- TCP의 6D Pose Error 세팅 ---------
        # Get the root & joint Pose in world frame
        root_pose_w = robot.data.root_state_w[:, :7]
        # Get the tcp pose in world frame
        lfinger_pose_w = robot.data.body_state_w[:, left_finger_link_idx, :7]
        rfinger_pose_w = robot.data.body_state_w[:, right_finger_link_idx, :7]
        lfinger_pos_b, lfinger_rot_b = subtract_frame_transforms(
            root_pose_w[:, :3], root_pose_w[:, 3:7], lfinger_pose_w[:, :3], lfinger_pose_w[:, 3:7])
        offset_vec_b = quat_apply(lfinger_rot_b, offset)

        tcp_loc_w = (lfinger_pose_w[:, :3] + rfinger_pose_w[:, :3]) / 2.0 + quat_apply(lfinger_pose_w[:, 3:7], torch.tensor([0.0, 0.0, 0.045], device=scene.device))
        tcp_pose_w = torch.cat((tcp_loc_w, lfinger_pose_w[:, 3:7]), dim=1)
        # Compute the tcp pose in the base frame
        tcp_loc_b, tcp_rot_b = subtract_frame_transforms(
                root_pose_w[:, :3], root_pose_w[:, 3:7], tcp_pose_w[:, :3], tcp_pose_w[:, 3:7])
        tcp_pose_b = torch.cat((tcp_loc_b, tcp_rot_b), dim=1)
        # Compute the position error in the base frame
        tcp_pos_err_b = tcp_pose_b[:, :3] - ik_commands[:, :3]
        tcp_rot_err_b = quat_error_magnitude(tcp_pose_b[:, 3:7], ik_commands[:, 3:])

        # --------- Target Points 갱신 로직 ---------
        if torch.norm(tcp_pos_err_b) < 5e-2 and tcp_rot_err_b < 5e-2:
            i = (i+1) %  optimal_trajectory.shape[0]
            if i == 0:
                # Trajectory 끝에 도착하면, Reset the Scene
                print("[INFO]: Resetting robot state...")
                root_state = robot.data.default_root_state.clone()
                root_state[:, :3] += scene.env_origins
                joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                robot.set_joint_effort_target(zero_joint_efforts, joint_ids=joint_ids)

                object_root_state = object.data.default_root_state.clone()
                object_root_state[:, :3] += scene.env_origins
                object.write_root_pose_to_sim(object_root_state[:, :7])
                object.write_root_velocity_to_sim(object_root_state[:, 7:])
                robot.write_data_to_sim()

                object.reset()
                robot.reset()
                scene.reset()
                robot.update(sim_dt)
            # IK Command를 Next Point로 갱신
            ik_commands[:, :] = optimal_trajectory[i, :]
            diff_ik_controller.set_command(ik_commands, tcp_pose_b[:, :3], tcp_pose_b[:, 3:7])

        # --------- Controller 동작 (IK) -------------
        # Compute Jacobian
        jacobian = robot.root_physx_view.get_jacobians()[:, jacobi_idx, :, robot_entity_cfg.joint_ids]
        # Compute Jacobian for TCP Point
        S_offset = compute_skew_symmetric_matrix(offset_vec_b)
        jacobian[:, :3, :] -= torch.bmm(S_offset, jacobian[:, 3:6, :])
        # Get the root & joint Pose in world frame
        joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
        joint_vel = robot.data.joint_vel[:, robot_entity_cfg.joint_ids]
        # Compute the desired joint position using the IK and the end-effector pose from the base frame
        joint_pos_des = diff_ik_controller.compute(tcp_loc_b, tcp_rot_b, jacobian, joint_pos)

        
        # --------- Controller 동작 (Impedance) ---------
        # Set command for Impedance control
        imp_commands[:, :7] = joint_pos_des - joint_pos
        joint_imp_controller.set_command(command=imp_commands)
        # Target torques 계산 (중력, 관성, 코리올리 힘 보상)
        tau = joint_imp_controller.compute(
            dof_pos      = joint_pos,
            dof_vel      = joint_vel,
            mass_matrix  = robot.root_physx_view.get_generalized_mass_matrices()[:, :num_active_joint, :num_active_joint],
            gravity= robot.root_physx_view.get_gravity_compensation_forces()[:, robot_entity_cfg.joint_ids]
        )
        # Torque 신호 내부 버퍼에 저장
        robot.set_joint_effort_target(tau, joint_ids=robot_entity_cfg.joint_ids)
        # 버퍼에 저장된 제어 신호 시뮬레이션에 모두 입력
        robot.write_data_to_sim()

        # 물리 엔진 스텝
        sim.step()
        scene.update(sim_dt)
        robot.update(sim_dt)

        lfinger_pose_w = robot.data.body_state_w[:, left_finger_link_idx, :7]
        rfinger_pose_w = robot.data.body_state_w[:, right_finger_link_idx, :7]
        tcp_pose_w = (lfinger_pose_w[:, :3] + rfinger_pose_w[:, :3]) / 2.0 + quat_apply(lfinger_pose_w[:, 3:7], offset)
        tcp_pos_w = torch.cat((tcp_pose_w, lfinger_pose_w[:, 3:7]), dim=1)

        tcp_marker.visualize(tcp_pos_w[:, :3], tcp_pos_w[:, 3:7])
        traj_marker.visualize(optimal_trajectory[:, :3] + scene.env_origins + robot.data.default_root_state[:, :3])
        goal_marker.visualize(object_pose_w[:, :3], object_pose_w[:, 3:7])
        

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=0.01)
    sim = SimulationContext(sim_cfg)
    sim_dt = sim.get_physics_dt()
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 4.0], [0.0, 0.0, 0.0])
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

def compute_skew_symmetric_matrix(vec: torch.Tensor) -> torch.Tensor:
    """
    주어진 3D 벡터 배치를 왜대칭 행렬 배치로 변환합니다.

    이 함수는 외적 연산(a x b)을 행렬 곱(S(a) * b)으로 변환하는 데 사용됩니다.

    Args:
        vec: (N, 3) 형태의 3D 벡터 텐서. N은 배치 크기입니다.

    Returns:
        (N, 3, 3) 형태의 왜대칭 행렬 텐서.
    """
    batch_size = vec.shape[0]
    device = vec.device
    dtype = vec.dtype

    S = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)

    S[:, 0, 1] = -vec[:, 2]
    S[:, 0, 2] =  vec[:, 1]
    S[:, 1, 0] =  vec[:, 2]
    S[:, 1, 2] = -vec[:, 0]
    S[:, 2, 0] = -vec[:, 1]
    S[:, 2, 1] =  vec[:, 0]

    return S


if __name__ == "__main__":
    main()
    simulation_app.close()

