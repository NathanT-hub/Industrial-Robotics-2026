"""
Shared move helper for all robots.

execute_move() drives any robot from its current pose (or a given start pose)
to a target pose using RMRC, and optionally carries a payload with it. The
payload can be any Swift object (or list of objects) with a .T pose, e.g. the
bearing parts from create_bearing(), a wheel, a finished assembly, etc.
"""
from spatialmath import SE3

from move_arm_rmrc import move_arm_rmrc


def _as_list(payload):
    """Accept a single object, a list/tuple of objects, or None."""
    if payload is None:
        return []
    if isinstance(payload, (list, tuple)):
        return list(payload)
    return [payload]


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

    for q in q_matrix:
        robot.q = q
        if parts:
            T_ee = robot.fkine(q)
            for part, offset in zip(parts, offsets):
                part.T = T_ee * offset
        env.step(dt)

    return q_matrix
