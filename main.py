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
from execute_move import execute_move
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
	bearing_x = -1.2  # centre of bearing_table
	bearing_parts = create_bearing(env, bearing_x, 0, surface_height, outer_diameter=0.08, inner_diameter=0.03, height= bearing_height)
	for part in bearing_parts: # For loop is to add both the outer and inner 'part'
		env.add(part)
	
	# Add DoBot6 Robot into environment
	robot = DoBot6(base=SE3.Trans(0, -0.60, 0.6))  # Had to comment this out for now; you're missing some of the qlim paramters and the progam won't run
	robot.add_to_env(env)

	# Add RS007N Robot into environment
	nathanBot = RS007N(base=SE3.Trans(-0.6, 0, 0.6))  # centre of arm2_table
	nathanBot.add_to_env(env)

	# Add reBot B601-DM Robot into environment
	# ryanBot = ReBotB601(base=SE3.Trans(0, 0.6, 0.6))
	# ryanBot.add_to_env(env)

	env.step(0.05)

#----------------------------------------------------------------
#					Collision Check Routine | RMRC Move (Test)
#----------------------------------------------------------------
	# --- Pick-and-place test: bearing_table -> assembly_bench, 
	q_start = [0.3, -0.5, 0.6, 0.2, -0.4, 0.1]
	nathanBot.q = q_start

	T_pick = SE3(bearing_x, 0, 0.70) * SE3.Rx(pi)   # above the bearing, gripper facing down
	T_place = SE3(0, 0, 0.70) * SE3.Rx(pi)     # above the workbench centre, gripper facing down

	# Waypoints: hover above the bearing, descend to grasp, lift, carry, descend to place, release, lift
	T_hover_pick  = SE3(bearing_x, 0, 0.75) * SE3.Rx(pi)
	T_grasp_pick  = SE3(bearing_x, 0, surface_height + bearing_height + 0.01) * SE3.Rx(pi)
	T_via         = SE3(-0.6, -0.6, 0.75) * SE3.Rx(pi)  # swing around the base -- a straight line bearing->bench passes through the robot
	T_hover_place = SE3(0, 0, 0.75) * SE3.Rx(pi)
	T_grasp_place = SE3(0, 0, surface_height + bearing_height + 0.01) * SE3.Rx(pi)

	# 1) Move down to the bearing (nothing carried)
	execute_move(env, nathanBot, T_hover_pick)
	execute_move(env, nathanBot, T_grasp_pick)

	# 2) Lift and carry across to the workbench -- bearing follows the end-effector
	execute_move(env, nathanBot, T_hover_pick, payload=bearing_parts)
	execute_move(env, nathanBot, T_via, payload=bearing_parts)
	execute_move(env, nathanBot, T_hover_place, payload=bearing_parts)
	execute_move(env, nathanBot, T_grasp_place, payload=bearing_parts)

	# 3) Release and retreat -- bearing stays where it was placed
	execute_move(env, nathanBot, T_hover_place)
	
	env.hold()


if __name__ == "__main__":
	main()
