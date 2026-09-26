# Import required libraries
import time
import swift

from spatialmath import SE3

# Import required files
from create_static_environment import create_static_environment

# Import robot models
# from RS007N import RS007N
from DoBot6 import DoBot6

def main():
	"""Start Swift and build the static project environment."""
	env = swift.Swift()
	env.launch(realtime=True)
	create_static_environment(env)
	
	# Add DoBot6 Robot into environment
	robot = DoBot6(base=SE3.Trans(0, -0.60, 0.6))
	robot.add_to_env(env)

	env.step(0.05)
	env.hold()

if __name__ == "__main__":
	main()
