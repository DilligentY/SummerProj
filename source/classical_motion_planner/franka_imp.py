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
from isaaclab_assets import FRANKA_PANDA_CFG, FRANKA_PANDA_HIGH_PD_CFG
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
    # robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot",
    #                                          init_state=ArticulationCfg.InitialStateCfg(
    #         pos=(0.0, 0.0, 1.05),
    #         joint_pos={
    #         "panda_joint1": 0.0,
    #         "panda_joint2": -0.569,
    #         "panda_joint3": 0.0,
    #         "panda_joint4": -2.810,
    #         "panda_joint5": 0.0,
    #         "panda_joint6": 3.037,
    #         "panda_joint7": 0.741,
    #         "panda_finger_joint.*": 0.04,
    #     },
    #     ),
    #     )
    
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
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    robot_entity_cfg.resolve(scene)
    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()
    
    # 7개 조인트만
    joint_ids = robot_entity_cfg.joint_ids
    n_j = len(joint_ids)
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

    # Relative Impedance Controller
    joint_imp_cfg = JointImpedanceControllerCfg(command_type="p_rel", 
                                                impedance_mode="variable",
                                                stiffness=300.0,
                                                damping_ratio=0.1,
                                                inertial_compensation=True, 
                                                gravity_compensation=True,
                                                stiffness_limits=(0, 1000))
    
    joint_imp_controller = JointImpedanceController(cfg=joint_imp_cfg, 
                                                    num_robots=scene.num_envs,
                                                    dof_pos_limits=robot.data.joint_pos_limits[:, :n_j],
                                                    device=scene.device)
    # Initial Configuration
    zero_joint_efforts = torch.zeros(scene.num_envs, n_j, device=sim.device)
    q_init   = robot.data.default_joint_pos.clone()
    q_dot_init = robot.data.default_joint_vel.clone()
    q_target = q_init[:, :n_j].clone()

    # Update Buffer
    robot.update(sim_dt)
    
    count = 0
    while simulation_app.is_running():
        # reset joint state to default
        if count % 500 == 0:
            print("[INFO] Reset state ...")
            default_joint_pos = robot.data.default_joint_pos.clone()
            default_joint_vel = robot.data.default_joint_vel.clone()
            # 강제로 조인트 워프 -> 초기 Configuration 세팅
            robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
            # 내부 버퍼에 토크 저장 -> 텔레포트가 아닌 제어 입력을 위한 명령 신호
            robot.set_joint_effort_target(zero_joint_efforts, joint_ids=joint_ids)
            # 버퍼에 쓰인 전체 데이터 복사
            robot.write_data_to_sim()
            robot.reset()
            robot.update(sim_dt)
        else:
            pass
        
        # 타겟 조인트 포지션 버퍼에 저장 -> 실제 액추에이터 PD제어기로 인해 토크 계산을 위함
        robot.set_joint_position_target(default_joint_pos[:, :n_j], joint_ids=robot_entity_cfg.joint_ids)
        # 세팅된 제어 명령 전부 실행 -> set으로 저장해둔 신호들이 모두 실행
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)
        scene.update(sim_dt)
        count += 1




def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 4.5], [0.0, 0.0, 0.0])
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
    # run the main function
    main()
    # close sim app
    simulation_app.close()