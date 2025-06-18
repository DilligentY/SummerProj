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
    
    joint_ids = robot_entity_cfg.joint_ids
    point_marker_cfg = CUBOID_MARKER_CFG.copy()
    point_marker_cfg.markers["cuboid"].size = (0.03, 0.03, 0.03)
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

    joint_imp_cfg = JointImpedanceControllerCfg(command_type="p_rel", 
                                                impedance_mode="variable",
                                                stiffness=300.0,
                                                damping_ratio=0.1,
                                                inertial_compensation=True, 
                                                gravity_compensation=True,
                                                stiffness_limits=(0, 1000))
    
    joint_imp_controller = JointImpedanceController(cfg=joint_imp_cfg, 
                                                    num_robots=scene.num_envs,
                                                    dof_pos_limits=robot.data.joint_pos_limits[:, :7],
                                                    device=scene.device)
    
    # ---------- 환경 준비 ----------
    n_j          = 7           # Franka: 9 (팔7+그리퍼2) 또는 7
    test_joint   = 3                          # 움직일 관절 번호
    step_size    = 0.0                       # [rad] 상대 목표
    sim_len      = 3.0                        # [s] 실험 길이
    Kp_val       = 100.0                       # Stiffness
    kp_table = torch.tensor([100, 800, 300, 200, 80, 100, 100], device=scene.device)
    zeta         = 0.1                        # Damping ratio(=가상 댐퍼 비율)
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
    q_dot_init = robot.data.default_joint_vel.clone()
    q_target = q_init[:, :n_j].clone()
    # q_target[:, test_joint] += step_size

    # --- 초기 한 틱 돌려 관성 행렬 준비 -----------------------------------
    robot.write_joint_state_to_sim(q_init, q_dot_init)
    robot.reset()

    # 목표값 만들기 (각 관절 frame Relative Position)
    q_des_rel = torch.zeros_like(q_target)

    # --- 루프 ------------------------------------------------------------
    num_step = int(sim_len / scene.physics_dt) + 1
    log_t = []
    log_q = torch.zeros([num_step, n_j], device=scene.device)
    log_q[0, :] = robot.data.default_joint_pos[:, :n_j]
    i = 1
    t = 0.0
    while t < sim_len:
        
        # 1) 상대 offset 계산
        joint_pos = robot.data.joint_pos[:, :n_j]
        q_des_rel =  q_target - joint_pos
        print(f"target joint : {q_target}")
        print(f"current joint : {joint_pos}")

        # 2) commands 채우기
        commands.zero_()
        commands[:, pos_slice]   = q_des_rel
        commands[:, stiff_slice] = kp_table           # 모든 관절 동일 Kp
        commands[:, damp_slice]  = zeta             # Crit.damping

        # 3) set_command → compute 순서
        joint_imp_controller.set_command(commands)
        tau = joint_imp_controller.compute(
            dof_pos      = robot.data.joint_pos[:, :n_j],
            dof_vel      = robot.data.joint_vel[:, :n_j],
            mass_matrix  = robot.root_physx_view.get_generalized_mass_matrices()[:, :n_j, :n_j],
            gravity= robot.root_physx_view.get_gravity_compensation_forces()[:, :n_j]
        )
        tau += robot.root_physx_view.get_coriolis_and_centrifugal_compensation_forces()[:, :n_j]

        robot.set_joint_effort_target(tau, joint_ids=joint_ids)
        # 4) 물리 스텝
        scene.write_data_to_sim(); sim.step(); scene.update(scene.physics_dt)
        t += scene.physics_dt

        # 5) 로그 저장
        log_t.append(t)
        log_q[i, :] = robot.data.joint_pos[:, :n_j]

    # --- 결과 플롯 -------------------------------------------------
    import matplotlib.pyplot as plt
    import numpy as np
    import math
    log_t_np = np.stack(log_t)
    log_q_np = log_q.cpu().numpy()

    n_cols = 3                                    # 한 줄에 최대 3장씩
    n_rows = math.ceil(n_j / n_cols)              # 필요한 줄 수
    fig, axes = plt.subplots(n_rows, n_cols,
                            figsize=(4*n_cols, 3*n_rows),
                            sharex=True)         # 시간축 공유
    axes = axes.flatten()                         # 1-D로 편리하게

    for i in range(n_j):
        ax = axes[i]
        ax.plot(log_t_np, log_q_np[:, i], label="actual")
        ax.axhline(joint_limits[0, i, 0].cpu(), ls="--", label="lower_limit", color='r')
        ax.axhline(joint_limits[0, i, 1].cpu(), ls="--", label="upper_limit", color='g')
        ax.axhline(q_target[0, i].cpu(),         ls="--", label="target", color="k")
        ax.set_title(f"Joint {i}")
        ax.set_ylabel("angle [rad]")
        if i // n_cols == n_rows - 1:            # 마지막 행에만 X라벨
            ax.set_xlabel("time [s]")
        # 범례는 첫 번째 서브플롯에만
        if i == 0:
            ax.legend(loc="best")
    fig.tight_layout()
    plt.show()



def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=0.01)
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