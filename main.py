# Import required libraries
import time
import swift
from spatialmath import SE3
from ir_support import RectangularPrism
import numpy as np
from math import pi
from spatialgeometry import Cuboid

# Import required files from GitHub Repo
from create_static_environment import create_static_environment, create_bearing, create_wheel, run_conveyor
from robot_move import robot_move
from move_arm_rmrc import move_arm_rmrc
from execute_move import execute_move
from check_collision import is_collision
from teach_gui import TeachPendant

# Import robot models
from DoBot6 import DoBot6

from Robots.RS007N import RS007N

from Robots.reBotB601 import ReBotB601



def main():
	"""Start Swift and build the static project environment."""
	env = swift.Swift()
	env.launch(realtime=True)
	create_static_environment(env)
	
	# Create the bearing on its own table, keep a direct handle to its parts (has to be created in main as it's not static)
	surface_height = 0.6
	bearing_height = 0.01
	bearing_x = -1.2  # centre of bearing_table
	bearing_parts = create_bearing(env, bearing_x, 0, surface_height, outer_diameter=0.032, inner_diameter=0.01, height= bearing_height)
	for part in bearing_parts: # For loop is to add both the outer and inner 'part'
		env.add(part)

	# Create the wheel (rim + tyre STL meshes) at the start of conveyor1
	# Conveyor runs along y from -2.0 to -0.4 at x = 0.42; belt (minus rollers) starts at y = -1.9
	wheel_x = 0.42
	wheel_y = -1.85  # belt start + tyre radius (0.04) + small margin
	wheel_parts = create_wheel(env, wheel_x, wheel_y, surface_height)
	for part in wheel_parts:
		env.add(part)
	
	# Add DoBot6 Robot into environment
	# Base offset toward the assembly-bench and conveyor edges of arm1_table so both are within reach (~0.44m)
	robot = DoBot6(base=SE3.Trans(0.10, -0.35, 0.6))
	# Waiting pose = ready pose: folded elbow-up with the tool pointing down, well clear of the
	# straight-up singularity at q = 0, so RMRC moves can start straight from here
	q_ready = np.radians([-150, -20, 150, -35, -90, 30])
	robot.q = q_ready
	robot.add_to_env(env)

	# Add RS007N Robot into environment
	nathanBot = RS007N(base=SE3.Trans(-0.6, 0, 0.6))  # centre of arm2_table
	nathanBot.add_to_env(env)

	# Add reBot B601-DM Robot into environment
	ryanBot = ReBotB601(base=SE3.Trans(0, 0.6, 0.6))
	ryanBot.add_to_env(env)

	env.step(0.05)

#----------------------------------------------------------------
#					Step 1 | Conveyor delivers the wheel to the DoBot6
#----------------------------------------------------------------
	# Run the conveyor (blocking) until the wheel reaches the pick point
	wheel_pick_y = -0.6
	run_conveyor(env, wheel_parts, wheel_y, wheel_pick_y)

#----------------------------------------------------------------
#					Step 2 | DoBot6 places the wheel at the assembly bench centre (origin)
#----------------------------------------------------------------
	# Waypoints: hover above the wheel, descend to grasp, lift, carry, descend to place, release, lift
	wheel_place_x, wheel_place_y = 0, 0
	# The wheel lies flat and its top is ~15mm above the surface; the DoBot6 Link6 mesh protrudes ~19mm
	# past the end-effector frame, so grasp ~20mm above the wheel top so the flange face rests on it
	wheel_grasp_z = surface_height + 0.0155 + 0.02
	T_hover_wheel = SE3(wheel_x, wheel_pick_y, 0.75) * SE3.Rx(pi)
	T_grasp_wheel = SE3(wheel_x, wheel_pick_y, wheel_grasp_z) * SE3.Rx(pi) # flange on top of the wheel
	T_hover_wheel_place = SE3(wheel_place_x, wheel_place_y, 0.75) * SE3.Rx(pi)
	T_grasp_wheel_place = SE3(wheel_place_x, wheel_place_y, wheel_grasp_z) * SE3.Rx(pi)

	# 1) Move from the ready pose down to the wheel (nothing carried)
	execute_move(env, robot, T_hover_wheel)
	execute_move(env, robot, T_grasp_wheel)

	# 2) Lift and carry across to the assembly bench -- wheel follows the end-effector
	execute_move(env, robot, T_hover_wheel, payload=wheel_parts)
	execute_move(env, robot, T_hover_wheel_place, payload=wheel_parts)
	execute_move(env, robot, T_grasp_wheel_place, payload=wheel_parts)

	# 3) Release, lift clear, then return to the waiting pose out of the way of the RS007N
	execute_move(env, robot, T_hover_wheel_place)
	execute_move(env, robot, robot.fkine(q_ready))

#----------------------------------------------------------------
#					Step 3 | RS007N installs the bearing into the wheel hub
#----------------------------------------------------------------
	q_start = [0.3, -0.5, 0.6, 0.2, -0.4, 0.1]
	nathanBot.q = q_start

	# Bearing drops into the rim's hub bore (32mm, same as the bearing OD) and sits flush with the wheel,
	# whose underside is 5mm above the bench (see create_wheel)
	wheel_base_z = surface_height + 0.005

	# Waypoints: hover above the bearing, descend to grasp, lift, carry, descend to install, release, lift
	T_hover_pick  = SE3(bearing_x, 0, 0.75) * SE3.Rx(pi)
	T_grasp_pick  = SE3(bearing_x, 0, surface_height + bearing_height + 0.01) * SE3.Rx(pi)
	T_via         = SE3(-0.6, -0.6, 0.75) * SE3.Rx(pi)  # swing around the base -- a straight line bearing->bench passes through the robot
	T_hover_place = SE3(wheel_place_x, wheel_place_y, 0.75) * SE3.Rx(pi)
	T_grasp_place = SE3(wheel_place_x, wheel_place_y, wheel_base_z + bearing_height + 0.01) * SE3.Rx(pi)

	# 1) Move down to the bearing (nothing carried)
	execute_move(env, nathanBot, T_hover_pick)
	execute_move(env, nathanBot, T_grasp_pick)

	# 2) Lift and carry across to the wheel -- bearing follows the end-effector
	execute_move(env, nathanBot, T_hover_pick, payload=bearing_parts)
	execute_move(env, nathanBot, T_via, payload=bearing_parts)
	execute_move(env, nathanBot, T_hover_place, payload=bearing_parts)
	execute_move(env, nathanBot, T_grasp_place, payload=bearing_parts)

	# 3) Release and retreat -- bearing stays installed in the wheel
	execute_move(env, nathanBot, T_hover_place)

#----------------------------------------------------------------
#					Step # | Hand over to the teach/jog pendant
#----------------------------------------------------------------
	# The pendant runs the main loop (steps Swift itself) until its window is closed, so it replaces env.hold()
	# Comment Out function call when not needed in main
	pendant = TeachPendant(env, {"DoBot6": robot, "RS007N": nathanBot, "reBot B601": ryanBot},
						   min_tool_z=surface_height)
	pendant.run()


if __name__ == "__main__":
	main()
