import numpy as np
from roboticstoolbox import ctraj
from spatialmath.base import vex


def move_arm_rmrc(robot, T1, T2, steps=50, delta_t=0.05, q0=None,
         min_manipulability=1e-3, max_damping=0.05, clip_to_qlim=True):
    """
    Move `robot`'s end-effector from pose T1 to T2 using Resolved Motion
    Rate Control, full 3D pose and any number of joints.

    Closed-loop: each step's velocity is computed from the robot's *actual*
    pose (fkine) to the next trajectory pose, so tracking errors from damping
    or joint-limit clipping are corrected rather than accumulated.
    """
    if q0 is None:
        q0 = robot.q

    traj = ctraj(T1, T2, steps)
    q_matrix = np.zeros((steps, robot.n))
    q0 = np.asarray(q0, dtype=float)
    if np.allclose(robot.fkine(q0).A, T1.A, atol=1e-4):
        # already at T1 -- start from q0 so the arm doesn't jump to another IK branch
        q_matrix[0, :] = q0
    else:
        sol = robot.ikine_LM(T1, q0=q0)
        if not sol.success:
            raise RuntimeError("RMRC: inverse kinematics failed to find a starting pose at T1")
        q_matrix[0, :] = sol.q

    for i in range(steps - 1):
        T_actual = robot.fkine(q_matrix[i, :])
        lin_vel = (traj[i + 1].t - T_actual.t) / delta_t
        ang_vel = vex((traj[i + 1].R - T_actual.R) / delta_t @ T_actual.R.T)
        xdot = np.concatenate([lin_vel, ang_vel])

        J = robot.jacob0(q_matrix[i, :])

        # damped least squares -- avoids qdot exploding near a singularity
        m = np.sqrt(max(np.linalg.det(J @ J.T), 0.0))
        damping = max_damping * (1 - m / min_manipulability) ** 2 if m < min_manipulability else 0.0
        q_dot = J.T @ np.linalg.inv(J @ J.T + (damping ** 2) * np.eye(6)) @ xdot

        q_matrix[i + 1, :] = q_matrix[i, :] + delta_t * q_dot
        if clip_to_qlim:
            qlim = robot.qlim
            q_matrix[i + 1, :] = np.clip(q_matrix[i + 1, :], qlim[0, :], qlim[1, :])

    return q_matrix