import numpy as np
from roboticstoolbox import ctraj
from spatialmath.base import vex


def move_arm_rmrc(robot, T1, T2, steps=50, delta_t=0.05, q0=None,
         min_manipulability=1e-3, max_damping=0.05, clip_to_qlim=True):
    """
    Move `robot`'s end-effector from pose T1 to T2 using Resolved Motion
    Rate Control, full 3D pose and any number of joints.

    Note on robot.jacob0(): roboticstoolbox's analytic Jacobian has a sign
    bug for any joint defined with flip=True (RS007N's joint 1). fkine() is
    unaffected -- verified independently against a finite-difference
    Jacobian. We correct for it below by flipping the sign of that joint's
    column, rather than avoiding jacob0() altogether.
    """
    if q0 is None:
        q0 = robot.q

    traj = ctraj(T1, T2, steps)
    q_matrix = np.zeros((steps, robot.n))
    sol = robot.ikine_LM(T1, q0=q0)
    if not sol.success:
        raise RuntimeError("RMRC: inverse kinematics failed to find a starting pose at T1")
    q_matrix[0, :] = sol.q

    # correction for roboticstoolbox's flip=True Jacobian sign bug
    flip_signs = np.array([-1.0 if link.isflip else 1.0 for link in robot.links])

    for i in range(steps - 1):
        lin_vel = (traj[i + 1].t - traj[i].t) / delta_t
        ang_vel = vex((traj[i + 1].R - traj[i].R) / delta_t @ traj[i].R.T)
        xdot = np.concatenate([lin_vel, ang_vel])

        J = robot.jacob0(q_matrix[i, :]) * flip_signs

        # damped least squares -- avoids qdot exploding near a singularity
        m = np.sqrt(max(np.linalg.det(J @ J.T), 0.0))
        damping = max_damping * (1 - m / min_manipulability) ** 2 if m < min_manipulability else 0.0
        q_dot = J.T @ np.linalg.inv(J @ J.T + (damping ** 2) * np.eye(6)) @ xdot

        q_matrix[i + 1, :] = q_matrix[i, :] + delta_t * q_dot
        if clip_to_qlim:
            qlim = robot.qlim
            q_matrix[i + 1, :] = np.clip(q_matrix[i + 1, :], qlim[0, :], qlim[1, :])

    return q_matrix