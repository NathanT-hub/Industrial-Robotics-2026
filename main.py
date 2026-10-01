# Import required libraries
import time
import swift
from spatialmath import SE3
from ir_support import RectangularPrism
import numpy as np

# Import required files from GitHub Repo
from create_static_environment import create_static_environment
from robot_move import robot_move
from move_arm_rmrc import move_arm_rmrc
from check_collision import is_collision

# Import robot models
from Robots.RS007N import RS007N

# from Robots.ReBotB601 import ReBotB601
#print(ReBotB601)

from DoBot6 import DoBot6

def main():
	"""Start Swift and build the static project environment."""
	env = swift.Swift()
	env.launch(realtime=True)
	create_static_environment(env)
	
	# Add DoBot6 Robot into environment
	robot = DoBot6(base=SE3.Trans(0, -0.60, 0.6))  # Had to comment this out for now; you're missing some of the qlim paramters and the progam won't run
	robot.add_to_env(env)

	# Add RS007N Robot into environment
	nathanBot = RS007N(base=SE3.Trans(-0.8, 0, 0.6))
	nathanBot.add_to_env(env)

	# Add reBot B601-DM Robot into environment
	# ryanBot = ReBotB601(base=SE3.Trans(0, 0.6, 0.6))
	# ryanBot.add_to_env(env)

	env.step(0.05)
	# path = robot_move(robot, [0.2, 0, 0.75])

	# for q in path.q:
	# 	robot.q = q
	# 	env.step(0.05)

	# --- deliberately set up a guaranteed collision, just to prove the check works ---
	q_start = [0.3, -0.5, 0.6, 0.2, -0.4, 0.1]
	nathanBot.q = q_start

	T1 = nathanBot.fkine(nathanBot.q)
	T2 = SE3(0.3, 0, 0) * T1   # move 30cm in world X

	# obstacle sitting right on the straight line between T1 and T2
	midpoint = (T1.t + T2.t) / 2
	vertices, faces, face_normals = RectangularPrism(0.15, 0.15, 0.15, center=midpoint).get_data()

	q_matrix = move_arm_rmrc(nathanBot, T1, T2)
	
	# --- TEMP DIAGNOSTIC: how close did any link actually get to the obstacle? ---
	min_dist = float('inf')
	for q in q_matrix:
		tr = nathanBot.fkine_all(q).A
		for i in range(len(tr) - 1):
			p1, p2 = tr[i][:3, 3], tr[i + 1][:3, 3]
			seg = p2 - p1
			t = np.clip(np.dot(midpoint - p1, seg) / np.dot(seg, seg), 0, 1)
			closest = p1 + t * seg
			d = np.linalg.norm(closest - midpoint)
			min_dist = min(min_dist, d)
	print("Closest any link ever got to the obstacle centre:", min_dist, "m")
	print("Obstacle half-diagonal (need to be closer than this to overlap):", (0.15 * np.sqrt(3)) / 2, "m")



	collision_spheres = []
	collided = False
	for q in q_matrix:
		if is_collision(nathanBot, [q], faces, vertices, face_normals, collision_spheres, env=env, return_once_found=True):
			print("Collision detected -- stopping nathanBot.")
			collided = True
			break
		nathanBot.q = q
		env.step(0.05)

	if not collided:
		print("No collision detected -- nathanBot completed the move.")

	env.hold()

if __name__ == "__main__":
	main()
