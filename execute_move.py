"""
Shared move helper for all robots.

execute_move() drives any robot from its current pose (or a given start pose)
to a target pose using RMRC, and optionally carries a payload with it. The
payload can be any Swift object (or list of objects) with a .T pose, e.g. the
bearing parts from create_bearing(), a wheel, a finished assembly, etc.

Collision response: call set_collision_guard() once in main(). While a move runs,
the rest of its planned trajectory is checked against the obstacles every few steps.
If an upcoming collision is predicted the arm holds where it is, a red marker is drawn
at the predicted contact, and on_blocked() is called (the pendant uses it to latch a
fault and wait for Resume). The move carries on, from the same step, once the path is clear.
"""
import numpy as np
from spatialmath import SE3

from move_arm_rmrc import move_arm_rmrc
from check_collision import is_collision

CHECK_EVERY = 3     # look ahead once every this many steps of a move
CHECK_STRIDE = 4    # ...testing every this-many'th remaining configuration (speed vs. resolution)

_guard = {"obstacles": [], "on_blocked": None}


def set_collision_guard(obstacles, on_blocked=None):
    """
    obstacles  -- list of objects with get_data() -> (vertices, faces, face_normals)
    on_blocked -- optional callback(message), called each time a blocked path is found
    """
    _guard["obstacles"] = list(obstacles)
    _guard["on_blocked"] = on_blocked


def _as_list(payload):
    """Accept a single object, a list/tuple of objects, or None."""
    if payload is None:
        return []
    if isinstance(payload, (list, tuple)):
        return list(payload)
    return [payload]


def _path_blocked(robot, q_remaining, env=None, markers=None):
    """True if any obstacle intersects the robot along the remaining configurations."""
    lookahead = np.vstack((q_remaining[::CHECK_STRIDE], q_remaining[-1]))
    for obstacle in _guard["obstacles"]:
        vertices, faces, normals = obstacle.get_data()
        if is_collision(robot, lookahead, faces, vertices, normals, collisions=markers, env=env):
            return True
    return False


def _hold_while_blocked(env, robot, q_remaining, dt):
    """Hold position (still rendering) until the remaining path is clear."""
    markers = []
    while _path_blocked(robot, q_remaining, env=None if markers else env, markers=markers):
        if _guard["on_blocked"] is not None:
            _guard["on_blocked"]("COLLISION PREDICTED - arm held. Clear the obstacle, then press Resume")
        env.step(dt)  # with the pendant attached, this freezes here until Resume is pressed
    for marker in markers:
        env.remove(marker)


def execute_move(env, robot, T_to, T_from=None, payload=None, steps=50, dt=0.05):
    """
    Move `robot`'s end-effector to `T_to`, optionally carrying `payload`.

    env     -- the Swift environment
    robot   -- any robot model (RS007N, DoBot6, reBotB601, ...)
    T_to    -- target end-effector pose (SE3)
    T_from  -- start pose (SE3); defaults to the robot's current pose
    payload -- object or list of objects to carry; each keeps the same
               relative pose to the end-effector it had when the move started
    steps   -- number of RMRC steps
    dt      -- time step passed to env.step()

    Returns the joint trajectory (steps x n) that was executed.
    """
    if T_from is None:
        T_from = robot.fkine(robot.q)

    parts = _as_list(payload)

    # "Grab": record each part's pose relative to the end-effector right now.
    # Done per part so multi-piece objects keep their own relative layout.
    T_ee_start = robot.fkine(robot.q)
    offsets = [T_ee_start.inv() * SE3(part.T) for part in parts]

    q_matrix = move_arm_rmrc(robot, T_from, T_to, steps=steps, q0=robot.q)

    for k, q in enumerate(q_matrix):
        # Look ahead along the rest of the planned path before taking the next step
        if _guard["obstacles"] and k % CHECK_EVERY == 0:
            _hold_while_blocked(env, robot, q_matrix[k:], dt)

        robot.q = q
        if parts:
            T_ee = robot.fkine(q)
            for part, offset in zip(parts, offsets):
                part.T = T_ee * offset
        env.step(dt)

    return q_matrix