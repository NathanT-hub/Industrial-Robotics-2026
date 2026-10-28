# File will generate all static items within the environment
# - Conveyor belts
# - Tables
# - Shelves
# - Safety markings / structures

from math import pi
from spatialmath import SE3
from spatialgeometry import Cuboid, Cylinder
from ir_support_extra_parts.parts import part_mesh
from spatialmath.base import trotz
import spatialgeometry as geometry
import os

# Constant values used by multiple functions
WHEEL_MESH_LIFT = 0.0004705665342044085
WHEEL_TOP_LOCAL_Z = 0.010403605177998543
FLOOR_GREY = [0.48, 0.50, 0.52, 1]
STEEL_BLUE = [0.12, 0.25, 0.38, 1]
LIGHT_METAL = [0.72, 0.75, 0.78, 1]
DARK_METAL = [0.20, 0.23, 0.26, 1]
SAFETY_YELLOW = [1, 0.78, 0.08, 1]
AISLE_GREEN = [0.16, 0.39, 0.30, 1]

def create_static_environment(env):
    # Add all static items to swift environment
    surface_height = 0.60
    # Create a dictionary to hold all static items as a list of their parts (Shapes)
    static_items = {}
    static_items["factory_details"] = create_factory_details()
    static_items["wallN1"] = create_wall(env, 2, 2, SE3.Trans(-1.4, 2.2, 0))
    static_items["wallN2"] = create_wall(env, 1.2, 2, SE3.Trans(1, 2.2, 0))
    static_items["wallW1"] = create_wall(env, 5.2, 2, SE3.Trans(-2.4, -0.4, 0)*SE3.Rz(pi/2))
    static_items["wallE1"] = create_wall(env, 5.2, 2, SE3.Trans(1.6, -0.4, 0)*SE3.Rz(pi/2))
    static_items["wallS1"] = create_wall(env, 2, 2, SE3.Trans(-1.4, -3, 0))
    static_items["assembly_bench"] = create_table(env, 0, 0, 0.40, 0.40, surface_height)
    static_items["arm1_table"] = create_table(env, 0, -0.45, 0.40, 0.40, surface_height) # 5cm gap to the assembly bench so DoBot6 can reach it
    static_items["arm2_table"] = create_table(env, -0.6, 0, 0.40, 0.40, surface_height)
    static_items["arm3_table"] = create_table(env, 0, 0.45, 0.40, 0.40, surface_height) # 5cm gap to the assembly bench so the reBot can reach it
    static_items["bearing_table"] = create_table(env, -1.2, 0, 0.40, 0.40, surface_height)
    static_items["conveyor1"] = create_conveyor(env, 1.6, 0.4, surface_height, SE3.Trans(0.42, -1.2, 0) * SE3.Rz(pi/2))
    static_items["dispatch_shelf"] = create_shelf(env, 0, 0.85, 0.40, 0.10, 0.60)
    # ---------------------------------------------------------------
    # Setting up workspace |safety features|
    # ---------------------------------------------------------------
    # South gate (y=-3, open side, closest to the camera)
    curtain_south_pose1 = SE3(-0.4, -3, 0.0)
    curtain_south_pose2 = SE3(1.6, -3, 0.0)
    # North gate (y=2.2, far side inspection door)
    curtain_north_pose1 = SE3(-0.4, 2.2, 0.0)
    curtain_north_pose2 = SE3(0.4, 2.2, 0.0)
    env.add(part_mesh("SafetyLightCurtain", pose=curtain_south_pose1, color="#e8491d"))
    env.add(part_mesh("SafetyLightCurtain", pose=curtain_south_pose2, color="#e8491d"))
    env.add(part_mesh("SafetyLightCurtain", pose=curtain_north_pose1, color="#e8491d"))
    env.add(part_mesh("SafetyLightCurtain", pose=curtain_north_pose2, color="#e8491d"))

    # # Button mounted on the made post
    estop_pose = SE3(-0.6, -3.12, 1.2) * SE3.Rz(pi)
    env.add(part_mesh("emergencyStopWallMounted", pose=estop_pose, color="#cc0000"))

    # SafetyPerson standing beside the post, offset sideways so they don't
    # overlap it, facing the same direction (toward the cell)
    safety_person_pose = SE3(0, -3.4, 0.0) * SE3.Rz(-pi/2)
    env.add(part_mesh("SafetyPerson", pose=safety_person_pose))

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

def create_factory_details():
    """
    Create visual factory features within the existing scenario, such as floor markings and safety barriers.
    """
    parts = []

    def box(size, position, colour):
        parts.append(Cuboid(
            scale=size,
            pose=SE3.Trans(*position),
            color=colour
        ))

    # Create floor
    floor_top_z = 0.002
    floor_thickness = 0.04
    box([4, 5.2, floor_thickness], [-0.4, -0.4, floor_top_z-floor_thickness/2], FLOOR_GREY)

    # Variables to ensure paint sits slightly above floor for good visibility and no flickering in Swift
    paint_thickness = 0.001
    surface_gap = 0.0005
    paint_z = floor_top_z + surface_gap + paint_thickness/2

    def floor_outline(cx, cy, length, width):
        # Rectangular equipment boundary using yellow strips
        line_width = 0.04

        # Horizontal edges
        for dy in [-width/2, width/2]:
            box(
                [length, line_width, paint_thickness],
                [cx, cy + dy, paint_z],
                SAFETY_YELLOW
            )

        # Vertical edges
        for dx in [-length/2, length/2]:
            box(
                [line_width, width, paint_thickness],
                [cx + dx, cy, paint_z],
                SAFETY_YELLOW
            )

    # Aisle
    aisle_x, aisle_y = 1.3, -0.4
    aisle_length, aisle_width = 4.8, 0.4
    edge_width = 0.025
    box([aisle_width, aisle_length, paint_thickness], [aisle_x, aisle_y, paint_z], AISLE_GREEN)
    # White edging alongside aisle
    edge_z = (paint_z + paint_thickness/2 + surface_gap + paint_thickness/2)
    for side in [-1, 1]:
        edge_x = aisle_x + side * (aisle_width/2 + edge_width/2)
        box([edge_width, aisle_length, paint_thickness], [edge_x, aisle_y, edge_z], [0.9, 0.92, 0.92, 1])

    floor_outline(cx=-0.3, cy=-0.35, length=2.6, width=3.6)  # Assembly bench area
    
    return parts
    
def create_table(env, x, y, length, width, height):
    tabletop_thickness = 0.04
    leg_radius = 0.04

    parts = []

    # Tabletop part dimensions length x width positioned with centre at x, y and top surface at z = height
    tabletop = Cuboid(
        scale = [length, width, tabletop_thickness],
        pose = SE3.Trans(x, y, height - tabletop_thickness/2),
        color = LIGHT_METAL
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
                color = STEEL_BLUE
            )
            parts.append(leg)

    # Create a visible frame underneath the tabletop to support it, using 4 cuboids to represent the beams
    frame_height = 0.06
    frame_width = 0.025
    frame_z = height - tabletop_thickness - frame_height/2

    # Front and rear frame beams
    for dy in [-width/2 + frame_width/2, width/2 - frame_width/2]:
        frame = Cuboid(
            scale = [length, frame_width, frame_height],
            pose = SE3.Trans(x, y + dy, frame_z),
            color = STEEL_BLUE
        )
        parts.append(frame)

    # Side frame beams
    for dx in [-length/2 + frame_width/2, length/2 - frame_width/2]:
        frame = Cuboid(
            scale = [frame_width, width, frame_height],
            pose = SE3.Trans(x + dx, y, frame_z),
            color = STEEL_BLUE
        )
        parts.append(frame)
    return parts

def create_shelf(env, x, y, length, width, height):
    thickness = 0.04
    parts = []

    # Shelf part dimensions length x width positioned with centre at x, y and top surface at z = height
    shelf = Cuboid(
        scale = [length, width, thickness],
        pose = SE3.Trans(x, y, height - thickness/2),
        color = LIGHT_METAL
    )
    parts.append(shelf)

    leg_height = height-thickness # Leg height based off shelf data
    x_offset = length / 2 - thickness / 2 # Values to position legs at each corner with the radius tangential to outer edge of shelf
    # Iterate through each corner of the shelf and generate a leg from the shelf to the ground
    for dx in [-x_offset, x_offset]:
        leg = Cuboid(
            scale = [thickness, width, leg_height],
            pose = SE3.Trans(x + dx, y, leg_height / 2),
            color = STEEL_BLUE
        )
        parts.append(leg)

    brace = Cuboid(
        scale = [length-2*thickness, thickness, thickness],
        pose = SE3.Trans(x, y, height/4),
        color = [0.6, 0.6, 0.65, 1]
    )
    parts.append(brace)
    return parts

def create_wall(env, length, height, base):
    thickness = 0.08

    wall = Cuboid(
        scale = [length, thickness, height],
        pose = base * SE3.Trans(0, -thickness / 2, height / 2),
        color = [0.6, 0.6, 0.65, 1]
    )
    return [wall]

def create_bearing(env, x, y, z, outer_diameter=0.032, inner_diameter=0.01, height=0.01):
    """
    Visual approximation of a bearing for pick-and-place: a solid outer
    ring (OD) with a darker inner cylinder (ID) to fake the bore -- no
    real hole is modelled, since nothing needs to pass through it.
    z is the surface it sits on (e.g. surface_height), not its centre.
    """
    parts = []

    outer = Cylinder(
        radius=outer_diameter / 2,
        length=height,
        pose=SE3.Trans(x, y, z + height / 2),  # centre offset so it sits ON the surface
        color=[0.75, 0.75, 0.78, 1]
    )
    parts.append(outer)

    # fake the bore -- slightly taller so it pokes through top and bottom
    # and doesn't z-fight with the outer cylinder's faces
    bore = Cylinder(
        radius=inner_diameter / 2,
        length=height * 1.05,
        pose=SE3.Trans(x, y, z + height / 2),
        color=[0.05, 0.05, 0.05, 1]
    )
    parts.append(bore)

    return parts
#-------------------------------------------------------------------
#--------------- Wheel STL file ------------------------------------
_MESH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wheel_meshes')

def create_wheel(env, x, y, z, yaw=0.0):
    rim_path = os.path.join(_MESH_DIR, 'wheel_rim.stl')
    tyre_path = os.path.join(_MESH_DIR, 'wheel_tyre.stl')

    pose = SE3.Trans(x, y, z + WHEEL_MESH_LIFT) * SE3.Rz(yaw)
    rim = geometry.Mesh(rim_path, color=[0.75, 0.75, 0.78, 1], pose=pose)
    tyre = geometry.Mesh(tyre_path, color=[0.08, 0.08, 0.08, 1], pose=pose)

    return [rim, tyre]

def create_conveyor(env, length, width, height, base):
    belt_color = (0.30, 0.32, 0.36, 1)
    frame_color = LIGHT_METAL

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

    # Four support legs beneath the conveyor, inset from the ends and sides
    support_height = height - rail_height
    for local_x in [-length / 2 + leg_inset, length / 2 - leg_inset]:
        for local_y in [-width / 2 + leg_inset, width / 2 - leg_inset]:
            leg = Cylinder(
                radius = leg_radius,
                length = support_height,
                pose = base * SE3.Trans(local_x, local_y, support_height / 2),
                color = STEEL_BLUE
            )
            parts.append(leg)

    return parts

"""
Blocking conveyor run: slides parts along world +y from start_y to stop_y, then stops
(acts like an end-stop sensor so the robot always knows where to pick from).
"""
def run_conveyor(env, parts, start_y, stop_y, speed=0.2, dt=0.05):
    y = start_y
    while y < stop_y:
        dy = min(speed * dt, stop_y - y) # Don't overshoot the stop point
        for part in parts:
            part.T = SE3.Trans(0, dy, 0) * SE3(part.T)
        y += dy
        env.step(dt)



    # Keep as example code if barrier needed later
    # barrier12_pose = SE3(-2.5, 1.739, 0.0) * SE3.Rz(pi / 2)
    # env.add(part_mesh("barrier1.5x0.2x1m", pose=barrier12_pose, color="#f2c14e"))

    # # Worker station: 1m outside the west wall (x=-2.5), near the gate,
    # # rotated to face the actual work (the table/UR3e at 0.6, 0.0)
    # worker_station_pose = SE3(-3.5, -2.5, 0.0) * SE3.Rz(pi)
    # estop_post_height = 2.0
    # estop_post_pose = worker_station_pose * SE3(0.15, 0, 1)
    # estop_post = Cuboid(scale=[0.15, 0.15, 2], pose=estop_post_pose, color="#555555")
    # env.add(estop_post)


