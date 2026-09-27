import roboticstoolbox as rtb
from spatialmath import SE3

def robot_move(robot, ee_pos, steps = 50):
    target = SE3.Trans(*ee_pos)
    q_start = robot.q.copy()
    q_end = robot.ikine_LM(target, q0=q_start, mask=[1,1,1,0,0,0]).q
    trajectory = rtb.jtraj(q_start, q_end, steps)
    return trajectory
    