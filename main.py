# Import required libraries
import time
import swift
from spatialmath import SE3

# Import required files from GitHub Repo
from create_static_environment import create_static_environment
from robot_move import robot_move
from move_arm_rmrc import move_arm_rmrc

# Import robot models
from Robots.RS007N import RS007N
print(RS007N)

#from Robots.ReBotB601 import ReBotB601
#print(ReBotB601)

# from DoBot6 import DoBot6

def main():
	"""Start Swift and build the static project environment."""
	env = swift.Swift()
	env.launch(realtime=True)
	create_static_environment(env)
	
	# Add DoBot6 Robot into environment
	# robot = DoBot6(base=SE3.Trans(0, -0.60, 0.6))   Had to comment this out for now; you're missing some of the qlim paramters and the progam won't run
	# robot.add_to_env(env)

	# Add DoBot6 Robot into environment
	nathanBot = RS007N(base=SE3.Trans(-0.6, 0, 0.6))
	nathanBot.add_to_env(env)

	# Add reBot B601-DM Robot into environment
	#ryanBot = ReBotB601(base=SE3.Trans(0, 0.6, 0.6))
	#ryanBot.add_to_env(env)

	env.step(0.05)
	# path = robot_move(robot, [0.2, 0, 0.75])

	# for q in path.q:
	# 	robot.q = q
	# 	env.step(0.05)

	env.hold()

if __name__ == "__main__":
	main()
