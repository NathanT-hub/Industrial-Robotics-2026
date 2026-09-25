# File will generate all static items within the environment
# - Conveyor belts
# - Tables
# - Shelves
# - Safety markings / structures

from ir_support_extra_parts.parts import part_mesh

def create_static_environment(env):
    # Add all static items to swift environment
    table = part_mesh("RobotTable", color=[0.55, 0.55, 0.55])
    env.add(table)
    return env