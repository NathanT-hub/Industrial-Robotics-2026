import swift
from spatialmath import SE3
from Robots.RS007N import RS007N
from move_arm_rmrc import move_arm_rmrc

env = swift.Swift()
env.launch(realtime=True)

robot = RS007N()
env.add(robot)
robot.add_to_env(env)

# start from a non-singular ("bent") pose rather than home_q, since home_q
# is a singularity for this arm -- move it there first
q_start = [0.3, -0.5, 0.6, 0.2, -0.4, 0.1]
robot.q = q_start

T1 = robot.fkine(robot.q)          # current pose
T2 = SE3(0.4, 0.5, -0.1) * T1     # target: 15cm/10cm/-10cm relative to current pose

q_matrix = move_arm_rmrc(robot, T1, T2)

for q in q_matrix:
    robot.q = q
    env.step(0.05)

env.hold()