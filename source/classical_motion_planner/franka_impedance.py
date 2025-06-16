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
        ),
        )



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
    joint_imp_cfg = JointImpedanceControllerCfg(command_type="p_rel", 
                                                impedance_mode="variable",
                                                stiffness=0.0,
                                                damping_ratio=0.0,
                                                inertial_compensation=True, 
                                                gravity_compensation=True)
    
    joint_imp_controller = JointImpedanceController(cfg=joint_imp_cfg, 
                                                    num_robots=scene.num_envs,
                                                    dof_pos_limits=robot.data.joint_pos_limits,
                                                    device=scene.device)
    
    # ---------- 환경 준비 ----------
    n_j          = robot.num_joints           # Franka: 9 (팔7+그리퍼2) 또는 7
    test_joint   = 3                          # 움직일 관절 번호
    step_size    = 0.30                       # [rad] 상대 목표
    sim_len      = 4.0                        # [s] 실험 길이
    Kp_val       = 20.0                      # 원하는 가상 스프링
    zeta         = 1.0                        # 감쇠비(=가상 댐퍼 비율)

    # ---------- 슬라이스 정의 ----------
    pos_slice       = slice(0, n_j)
    stiff_slice     = slice(n_j, 2*n_j)
    damp_slice      = slice(2*n_j, 3*n_j)

    # ---------- 명령 버퍼 ----------
    commands = torch.zeros(scene.num_envs,
                        joint_imp_controller.num_actions,
                        device=scene.device)


    # --- 목표값 설정 -----------------------------------------------------
    robot.reset()
    q_home   = robot.data.default_joint_pos.clone()
    q_dot = robot.data.default_joint_vel.clone()
    q_target = q_home.clone()
    q_target[:, test_joint] += step_size

    # --- 초기 한 틱 돌려 관성 행렬 준비 -----------------------------------
    robot.write_joint_state_to_sim(q_home, q_dot)
    scene.write_data_to_sim(); sim.step(); scene.update(scene.physics_dt)

    # 목표값 만들기 (각 관절 frame 상대각)
    q_des_rel = torch.zeros_like(q_home)

    # --- 루프 ------------------------------------------------------------
    log_t, log_q = [], []
    t = 0.0
    while t < sim_len:
        
        # 1) 상대 offset 계산
        joint_pos = robot.data.joint_pos
        q_des_rel[:, test_joint] = q_target[:, test_joint] - joint_pos[:, test_joint]
        print(f"offset : {q_des_rel[:, test_joint]}")

        # 2) commands 채우기
        commands.zero_()
        commands[:, pos_slice]   = q_des_rel
        commands[:, stiff_slice] = Kp_val           # 모든 관절 동일 Kp
        commands[:, damp_slice]  = zeta  # Crit.damping

        # 3) set_command → compute 순서
        joint_imp_controller.set_command(commands)
        tau = joint_imp_controller.compute(
            dof_pos      = robot.data.joint_pos,
            dof_vel      = robot.data.joint_vel,
            mass_matrix  = robot.root_physx_view.get_generalized_mass_matrices(),
            gravity= robot.root_physx_view.get_gravity_compensation_forces()
        )

        robot.set_joint_effort_target(tau, joint_ids=joint_ids)

        # 4) 물리 스텝
        scene.write_data_to_sim(); sim.step(); scene.update(scene.physics_dt)
        t += scene.physics_dt

        # 5) 로그
        log_t.append(t)
        log_q.append(robot.data.joint_pos[0, test_joint].item())

    # --- 결과 플롯 (선택) -------------------------------------------------
    import matplotlib.pyplot as plt
    plt.plot(log_t, log_q, label="actual")
    plt.axhline(q_target[0, test_joint].cpu(), ls="--", label="target")
    plt.xlabel("time [s]"); plt.ylabel("joint angle [rad]")
    plt.title("Step response: joint {}".format(test_joint))
    plt.legend(); plt.show()

#    # ---- 상수 정의 ----
#     amp_val = 0.2                            # 진폭 [rad]  (조인트마다 다르게 쓰고 싶으면 벡터로)
#     T            = 1.0 / 0.1         # 한 사이클 길이
#     zero_vel     = torch.zeros_like(q_home)
#     init_rel     = torch.zeros_like(q_home)
#     t            = 0.0
#     last_cycle_i = -1                  # 직전에 실행한 조인트 인덱스
#     n_j     = robot.num_joints

#     # 루프 전에 한 번만 선언
#     commands = torch.zeros(scene.num_envs, joint_imp_controller.num_actions,
#                         device=scene.device)
#     kp  = 30.0
#     rho = 5.0
#     pos_slice       = slice(0, n_j)
#     stiffness_slice = slice(n_j, 2*n_j)
#     damping_slice   = slice(2*n_j, 3*n_j)

#     t = 0.0
#     while simulation_app.is_running():

#         # --- 0) 새 사이클을 시작해야 하나? ---------------------------
#         cycle_i = int(t // T)          # 몇 번째 사이클인지 (0,1,2,…)
#         if cycle_i != last_cycle_i:    # 막 새로 시작했다면
#             print(f"[INFO] start cycle {cycle_i}  (joint {cycle_i % n_j})")

#             # (a) 로봇 상태 초기화
#             q_des = init_rel.clone()
#             joint_pos = robot.data.default_joint_pos
#             joint_vel = robot.data.default_joint_vel
#             robot.write_joint_state_to_sim(joint_pos, joint_vel)
#             scene.write_data_to_sim()
#             sim.step()                 # ➊ reset 상태가 실제 시뮬에 반영되도록 한 틱 돌림
#             scene.update(sim_dt)


#             last_cycle_i = cycle_i           # 갱신
#             # 이후 계산은 평소처럼 계속 진행

#         # --- 1) 이번 사이클에서 움직일 조인트 -------------------------
#         active_joint = cycle_i % n_j

#         # --- 2) 원하는 각도(q_des) 계산 --------------------------------
#         phase = torch.tensor(2 * torch.pi * (t % T) / T, device=scene.device)   # 0 ~ 2π
#         q_des[:, active_joint] = amp_val * torch.sin(phase)
#         print("------------------------------------------")
#         print(f"desred val : {q_des[:, active_joint]}")

#         # --- 3) 컨트롤러 명령 ------------------------------------------
#         commands.zero_()
#         commands[:, pos_slice]       = q_des
#         commands[:, stiffness_slice] = kp
#         commands[:, damping_slice]   = rho
#         joint_imp_controller.set_command(commands)

#         # --- 4) 토크 계산 & 적용 ---------------------------------------
#         torque = joint_imp_controller.compute(
#             dof_pos = robot.data.joint_pos,
#             dof_vel = robot.data.joint_vel,
#             mass_matrix = robot.root_physx_view.get_generalized_mass_matrices()
#         )
#         robot.set_joint_effort_target(torque, joint_ids=joint_ids)

#         # --- 5) 시뮬레이터 스텝 ---------------------------------------
#         scene.write_data_to_sim()
#         sim.step()
#         scene.update(sim_dt)
#         t += sim_dt
#         print(f"current_val : {robot.data.joint_pos[:, active_joint]}")
#         print("---------------------------------------------")



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