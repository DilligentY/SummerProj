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

    joint_imp_cfg = JointImpedanceControllerCfg(command_type="p_rel", 
                                                impedance_mode="variable",
                                                stiffness=400.0,
                                                damping_ratio=1.0,
                                                inertial_compensation=True, 
                                                gravity_compensation=True)
    
    joint_imp_controller = JointImpedanceController(cfg=joint_imp_cfg, 
                                                    num_robots=scene.num_envs,
                                                    dof_pos_limits=robot.data.joint_pos_limits,
                                                    device=scene.device)
    
    # ---------- 환경 준비 ----------
    n_j          = robot.num_joints           # Franka: 9 (팔7+그리퍼2) 또는 7
    test_joint   = 1                          # 움직일 관절 번호
    step_size    = 1.0                       # [rad] 상대 목표
    sim_len      = 4.0                        # [s] 실험 길이
    Kp_val       = 50.0                       # Stiffness
    zeta         = 0.5                        # Damping ratio(=가상 댐퍼 비율)
    joint_limits = robot.data.joint_limits

    # ---------- 슬라이스 정의 ----------
    pos_slice       = slice(0, n_j)
    stiff_slice     = slice(n_j, 2*n_j)
    damp_slice      = slice(2*n_j, 3*n_j)

    # ---------- 명령 버퍼 ----------
    commands = torch.zeros(scene.num_envs,
                        joint_imp_controller.num_actions,
                        device=scene.device)


    # --- 목표값 설정 -----------------------------------------------------
    q_init   = robot.data.default_joint_pos.clone()
    q_dot = robot.data.default_joint_vel.clone()
    q_target = q_init.clone()
    q_target[:, test_joint] += step_size

    # --- 초기 한 틱 돌려 관성 행렬 준비 -----------------------------------
    robot.write_joint_state_to_sim(q_init, q_dot)
    robot.reset()

    # 목표값 만들기 (각 관절 frame Relative Position)
    q_des_rel = torch.zeros_like(q_init)

    # --- 루프 ------------------------------------------------------------
    log_t, log_q = [], []
    t = 0.0
    while t < sim_len:
        
        # 1) 상대 offset 계산
        joint_pos = robot.data.joint_pos
        q_des_rel[:, test_joint] =  q_target[:, test_joint] - joint_pos[:, test_joint]
        print(f"current joint : {joint_pos[:, test_joint]}")
        print(f"offset : {q_des_rel[:, test_joint]}")

        # 2) commands 채우기
        commands.zero_()
        commands[:, pos_slice]   = q_des_rel
        commands[:, stiff_slice] = Kp_val           # 모든 관절 동일 Kp
        commands[:, damp_slice]  = zeta             # Crit.damping

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

        # 5) 로그 저장
        log_t.append(t)
        log_q.append(robot.data.joint_pos[0, test_joint].item())

    # --- 결과 플롯 -------------------------------------------------
    import matplotlib.pyplot as plt
    plt.plot(log_t, log_q, label="actual")
    plt.axhline(joint_limits[0, test_joint, 0].cpu(), ls="--", label="lower_limit", color='r')
    plt.axhline(joint_limits[0, test_joint, 1].cpu(), ls="--", label="upper_limit", color='g')
    plt.axhline(q_target[0, test_joint].cpu(), ls="--", label="target", color='k')
    plt.xlabel("time [s]"); plt.ylabel("joint angle [rad]")
    plt.title("Step response: joint {}".format(test_joint))
    plt.legend(); plt.show()



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