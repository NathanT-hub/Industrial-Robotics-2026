# Import required libraries
import time
import swift
from spatialmath import SE3
from ir_support import RectangularPrism
import numpy as np
from math import pi
from spatialgeometry import Cuboid

# Import required files from GitHub Repo
from create_static_environment import create_static_environment
from create_static_environment import create_static_environment, create_bearing
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
	
	# Create the bearing on its own table, keep a direct handle to its parts (has to be created in main as it's not static)
	surface_height = 0.6
	bearing_height = 0.02
	bearing_parts = create_bearing(env, -1.4, 0, 0.6, outer_diameter=0.08, inner_diameter=0.03, height= bearing_height)
	for part in bearing_parts: # For loop is to add both the outer and inner 'part'
		env.add(part)
	
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

#----------------------------------------------------------------
#					Collision Check Routine | RMRC Move (Test)
#----------------------------------------------------------------
	# --- Pick-and-place test: bearing_table -> assembly_bench, with an obstacle in the way ---
	q_start = [0.3, -0.5, 0.6, 0.2, -0.4, 0.1]
	nathanBot.q = q_start

	T_pick = SE3(-1.4, 0, 0.70) * SE3.Rx(pi)   # above the bearing, gripper facing down
	T_place = SE3(0, 0, 0.70) * SE3.Rx(pi)     # above the workbench centre, gripper facing down

	# Waypoints: hover above the bearing, descend to grasp, lift, carry, descend to place, release, lift
	T_hover_pick  = SE3(-1.4, 0, 0.75) * SE3.Rx(pi)
	T_grasp_pick  = SE3(-1.4, 0, surface_height + bearing_height + 0.01) * SE3.Rx(pi)
	T_hover_place = SE3(0, 0, 0.75) * SE3.Rx(pi)
	T_grasp_place = SE3(0, 0, surface_height + bearing_height + 0.01) * SE3.Rx(pi)

	def run_leg(T_from, T_to, carrying=False, grasp_offset=None):
		q_matrix = move_arm_rmrc(nathanBot, T_from, T_to)
		for q in q_matrix:
			nathanBot.q = q
			if carrying:
				gripper_pose = nathanBot.fkine(nathanBot.q)
				bearing_pose = gripper_pose * grasp_offset
				for part in bearing_parts:
					part.T = bearing_pose
			env.step(0.05)

	# 1) Move down to the bearing
	run_leg(nathanBot.fkine(nathanBot.q), T_hover_pick)
	run_leg(T_hover_pick, T_grasp_pick)

	# 2) "Grab" -- record the fixed offset between gripper and bearing right now
	gripper_at_grasp = nathanBot.fkine(nathanBot.q)
	bearing_pose_at_grasp = SE3.Trans(-1.4, 0, surface_height + bearing_height / 2)
	grasp_offset = gripper_at_grasp.inv() * bearing_pose_at_grasp

	# 3) Lift and carry across to the workbench, bearing follows the gripper
	run_leg(T_grasp_pick, T_hover_pick, carrying=True, grasp_offset=grasp_offset)
	run_leg(T_hover_pick, T_hover_place, carrying=True, grasp_offset=grasp_offset)
	run_leg(T_hover_place, T_grasp_place, carrying=True, grasp_offset=grasp_offset)

	# 4) "Release" -- snap the bearing to its final resting pose on the table surface
	final_pose = SE3.Trans(0, 0, surface_height + bearing_height / 2)
	for part in bearing_parts:
		part.T = final_pose

	# 5) Retreat, bearing stays put (carrying=False)
	run_leg(T_grasp_place, T_hover_place)
	
	env.hold()


if __name__ == "__main__":
	main()
