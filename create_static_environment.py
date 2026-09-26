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
    surface_height = 0.60
    # Create a dictionary to hold all static items as a list of their parts (Shapes)
    static_items = {}
    static_items["assembly_bench"] = create_table(env, 0, 0, 0.40, 0.40, surface_height)
    static_items["arm1_table"] = create_table(env, 0, -0.6, 0.40, 0.40, surface_height)
    static_items["arm2_table"] = create_table(env, -0.6, 0, 0.40, 0.40, surface_height)
    static_items["arm3_table"] = create_table(env, 0, 0.6, 0.40, 0.40, surface_height)

    static_items["conveyor1"] = create_conveyor(env, 1.6, 0.4, surface_height, SE3.Trans(0.42, -1.2, 0) * SE3.Rz(pi/2))

    # Iterate through each list within the dictionary
    for item in static_items.values():
        # Iterate through each shape within the items list
        for part in item:
            env.add(part) # Add every individual part to the environment
    return env

"""
Method to create a table with centre at x, y and dimensions length, width.
Tabletop surface is at z = height
Returns a list containing 5 parts - 1 cuboid tabletop and 4 cylinder legs
"""
def create_table(env, x, y, length, width, height):
    tabletop_thickness = 0.04
    leg_radius = 0.04

    parts = []

    # Tabletop part dimensions length x width positioned with centre at x, y and top surface at z = height
    tabletop = Cuboid(
        scale = [length, width, tabletop_thickness],
        pose = SE3.Trans(x, y, height - tabletop_thickness/2)
        # colour = [0.6, 0.6, 0.65, 1]
    )
    parts.append(tabletop)

    leg_height = height-tabletop_thickness # Leg height based off tabletop data
    # Values to position legs at each corner with the radius tangential to outer edge of tabletop
    x_offset = length / 2 - leg_radius
    y_offset = width / 2 - leg_radius

    # Iterate through each corner of the tabletop and generate a leg from the tabletop to the ground
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

def create_conveyor(env, length, width, height, base):
    belt_color = (0.08, 0.08, 0.08, 1)
    frame_color = (0.55, 0.55, 0.60, 1)

    # Conveyor belt constants
    belt_thickness = 0.02
    rail_thickness = 0.02
    rail_height = 0.1
    leg_radius = 0.04
    roller_radius = 0.05

    leg_inset = 0.1

    parts = []

    # Create top belt surface
    belt_length = length - 2 * roller_radius # Shorten belt to account for rollers
    belt = Cuboid(
        scale = [belt_length, width, belt_thickness],
        pose = base*SE3.Trans(0, 0, height - belt_thickness / 2),
        color = belt_color
    )

    # Create 2 x side rails
    rail_y_offset = width / 2 + rail_thickness / 2
    rail_z = height - rail_height / 2

    for y in [-rail_y_offset, rail_y_offset]:
        rail = Cuboid(
            scale = [length, rail_thickness, rail_height],
            pose = base * SE3.Trans(0, y, rail_z),
            color = frame_color
        )
        parts.append(rail)

    # Create 2 x end rollers
    roller_x_offset = length / 2 - roller_radius
    roller_z = height - roller_radius

    for x in [-roller_x_offset, roller_x_offset]:
        roller = Cylinder(
            radius = roller_radius,
            length = width,
            pose = base * SE3.Trans(x, 0, roller_z) * SE3.Rx(pi/2),
            color = frame_color
        )
        parts.append(roller)

    parts.append(belt)



    return parts





