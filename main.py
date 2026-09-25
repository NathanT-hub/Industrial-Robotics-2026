# Import required libraries
import time
import swift

# Import required files
from create_static_environment import create_static_environment

def main():
	"""Start Swift and build the static project environment."""
	env = swift.Swift()
	env.launch(realtime=True)
	create_static_environment(env)
	env.hold()

if __name__ == "__main__":
	main()
