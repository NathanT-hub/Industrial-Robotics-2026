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

def create_static_environment(env):
    # Add all static items to swift environment
    surface_height = 0.60
    # Create a dictionary to hold all static items as a list of their parts (Shapes)
    static_items = {}
    static_items["assembly_bench"] = create_table(env, 0, 0, 0.40, 0.40, surface_height)
    static_items["arm1_table"] = create_table(env, 0, -0.45, 0.40, 0.40, surface_height) # 5cm gap to the assembly bench so DoBot6 can reach it
    static_items["arm2_table"] = create_table(env, -0.6, 0, 0.40, 0.40, surface_height)
    static_items["arm3_table"] = create_table(env, 0, 0.6, 0.40, 0.40, surface_height)
    static_items["bearing_table"] = create_table(env, -1.2, 0, 0.40, 0.40, surface_height)
    static_items["dispatch_table"] = create_table(env, 0, 1.2, 0.40, 0.40, surface_height)

    static_items["conveyor1"] = create_conveyor(env, 1.6, 0.4, surface_height, SE3.Trans(0.42, -1.2, 0) * SE3.Rz(pi/2))


    # ---------------------------------------------------------------
    # Setting up workspace |safety features|
    # ---------------------------------------------------------------
    # North wall (y=2.5), running along x
    barrier1_pose = SE3(-1.739, 2.5, 0.0)
    barrier2_pose = SE3(-0.580, 2.5, 0.0)
    barrier3_pose = SE3(0.580, 2.5, 0.0)
    barrier4_pose = SE3(1.739, 2.5, 0.0)

    # East wall (x=2.5), running along y - rotated 90 degrees
    barrier5_pose = SE3(2.5, -1.739, 0.0) * SE3.Rz(pi / 2)
    barrier6_pose = SE3(2.5, -0.580, 0.0) * SE3.Rz(pi / 2)
    barrier7_pose = SE3(2.5, 0.580, 0.0) * SE3.Rz(pi / 2)
    barrier8_pose = SE3(2.5, 1.739, 0.0) * SE3.Rz(pi / 2)

    # West wall (x=-2.5), running along y - rotated 90 degrees
    barrier9_pose = SE3(-2.5, -1.739, 0.0) * SE3.Rz(pi / 2)
    barrier10_pose = SE3(-2.5, -0.580, 0.0) * SE3.Rz(pi / 2)
    barrier11_pose = SE3(-2.5, 0.580, 0.0) * SE3.Rz(pi / 2)
    barrier12_pose = SE3(-2.5, 1.739, 0.0) * SE3.Rz(pi / 2)

    env.add(part_mesh("barrier1.5x0.2x1m", pose=barrier1_pose, color="#f2c14e"))
    env.add(part_mesh("barrier1.5x0.2x1m", pose=barrier2_pose, color="#f2c14e"))
    env.add(part_mesh("barrier1.5x0.2x1m", pose=barrier3_pose, color="#f2c14e"))
    env.add(part_mesh("barrier1.5x0.2x1m", pose=barrier4_pose, color="#f2c14e"))
    env.add(part_mesh("barrier1.5x0.2x1m", pose=barrier5_pose, color="#f2c14e"))
    env.add(part_mesh("barrier1.5x0.2x1m", pose=barrier6_pose, color="#f2c14e"))
    env.add(part_mesh("barrier1.5x0.2x1m", pose=barrier7_pose, color="#f2c14e"))
    env.add(part_mesh("barrier1.5x0.2x1m", pose=barrier8_pose, color="#f2c14e"))
    env.add(part_mesh("barrier1.5x0.2x1m", pose=barrier9_pose, color="#f2c14e"))
    env.add(part_mesh("barrier1.5x0.2x1m", pose=barrier10_pose, color="#f2c14e"))
    env.add(part_mesh("barrier1.5x0.2x1m", pose=barrier11_pose, color="#f2c14e"))
    env.add(part_mesh("barrier1.5x0.2x1m", pose=barrier12_pose, color="#f2c14e"))

    # South gate (y=-2.5, open side, closest to the camera)
    light_curtain_left_pose = SE3(-2.5, -2.5, 0.0)
    light_curtain_right_pose = SE3(2.5, -2.5, 0.0)

    env.add(part_mesh("SafetyLightCurtain", pose=light_curtain_left_pose, color="#e8491d"))
    env.add(part_mesh("SafetyLightCurtain", pose=light_curtain_right_pose, color="#e8491d"))

    # Worker station: 1m outside the west wall (x=-2.5), near the gate,
    # rotated to face the actual work (the table/UR3e at 0.6, 0.0)
    worker_station_pose = SE3(-3.5, -2.5, 0.0) *trotz(pi)
    estop_post_height = 2.0
    estop_post_pose = worker_station_pose * SE3(0.15, 0, 1)
    estop_post = Cuboid(scale=[0.15, 0.15, 2], pose=estop_post_pose, color="#555555")
    env.add(estop_post)

    # Button mounted on the made post
    estop_pose = worker_station_pose * SE3(0, 0, estop_post_height/2) 
    env.add(part_mesh("emergencyStopWallMounted", pose=estop_pose, color="#cc0000"))

    # SafetyPerson standing beside the post, offset sideways so they don't
    # overlap it, facing the same direction (toward the cell)
    safety_person_pose = worker_station_pose * SE3(0, 0.4, 0.0)
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

    pose = SE3.Trans(x, y, z + 0.005) * SE3.Rz(yaw)
    rim = geometry.Mesh(rim_path, color=[0.75, 0.75, 0.78, 1], pose=pose)
    tyre = geometry.Mesh(tyre_path, color=[0.08, 0.08, 0.08, 1], pose=pose)

    return [rim, tyre]

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





