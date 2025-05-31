# AISL/source/classical_motion_planner/nominal_test_rrt.py

# Top-Level Directory : source/classical_motion_planner
import torch
from utils import Env
from utils import Node
from RRT.RRT_star import RRTStar
from RRT.RRT_wrapper import RRTWrapper


# obs_rect = [
#     [18, 22, 5, 8, 3, 4],
#     [24, 20, 5, 8, 8, 4],
#     [26,  7, 5, 2, 12, 3],
#     [32, 14, 5, 10, 2, 8]
# ]


environment = Env.Map3D(5, 5, 5)
# environment.update(obs_rect=obs_rect)

planner_torch = RRTWrapper(
    start=torch.tensor([0.1, 0, 0.1, 0.707, 0, 0.707, 0], dtype=torch.float32),
    goal=torch.tensor([0.3, 0.3, 0.5, 0, 0.707, 0, 0.707], dtype=torch.float32),
    env=environment
)

ik_commands = torch.zeros(10, 7)
ee_goals = [0.5, 0.5, 0.7, 0.707, 0, 0.707, 0]
ee_goals = torch.tensor(ee_goals)


# planner = RRTStar(start = (18,8,2), goal=(35,22,10), env=environment)

try:
    path = planner_torch.plan()
    # planner_torch.run()
finally:
    print("End of RRT*")