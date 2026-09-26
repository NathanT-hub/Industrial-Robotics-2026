# File will generate all static items within the environment
# - Conveyor belts
# - Tables
# - Shelves
# - Safety markings / structures

from math import pi
from spatialmath import SE3
from spatialgeometry import Cuboid, Cylinder
from ir_support_extra_parts.parts import part_mesh

def create_static_environment(env):
    # Add all static items to swift environment
    table = part_mesh("RobotTable", color=[0.55, 0.55, 0.55])
    env.add(table)

    table1 = create_table(env, 1, 1, 0.40, 0.40, 0.50)
    for part in table1:
        env.add(part)

    return env

"""
Method to create a table with centre at x, y and dimensions length, width.
Tabletop surface is at z = height
"""
def create_table(env, x, y, length, width, height):
    tabletop_thickness = 0.04
    leg_radius = 0.04

    parts = []

    tabletop = Cuboid(
        scale = [length, width, tabletop_thickness],
        pose = SE3.Trans(x, y, height - tabletop_thickness/2)
        # colour = [0.6, 0.6, 0.65, 1]
    )
    parts.append(tabletop)

    leg_height = height-tabletop_thickness

    x_offset = length / 2 - leg_radius
    y_offset = width / 2 - leg_radius

    for dx in [-x_offset, x_offset]:
        for dy in [-y_offset, y_offset]:
            leg = Cylinder(
                radius = leg_radius,
                length = leg_height,
                pose = SE3.Trans(x + dx, y + dy, leg_height / 2),
                color = [0.6, 0.6, 0.65, 1]
            )
            parts.append(leg)

    return parts



