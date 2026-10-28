import numpy as np
from roboticstoolbox import ctraj
from scipy.spatial.transform import Rotation as R

def move_arm_rmrc(robot, T1, T2, steps=150, delta_t=0.05, q0=None,
         min_manipulability=1e-3, max_damping=0.05, position_tolerance=0.001, orientation_tolerance=np.radians(1)):
    """
    Move `robot`'s end-effector from pose T1 to T2 using Resolved Motion
    Rate Control, full 3D pose and any number of joints.
    Closed-loop: each step's velocity is computed from the robot's *actual*
    pose (fkine) to the next trajectory pose, so tracking errors from damping
    or joint speeds caling are corrected at subsequent steps. Joint limit violations
    are rejected rather than clipped
    """
    if steps < 2 or not np.isfinite(delta_t) or delta_t <= 0:
        raise ValueError("RMRC requires at least 2 steps and a positive delta_t")

    if q0 is None:
        q0 = robot.q
    traj = ctraj(T1, T2, steps)
    q_matrix = np.zeros((steps, robot.n))
    
    q0 = np.asarray(q0, dtype=float)
    qlim=robot.qlim
    # Additional error checks
    if q0.shape != (robot.n,) or not np.isfinite(q0).all():
        raise ValueError("RMRC received an invalid starting joint vector")
    if np.any(q0 < qlim[0]) or np.any(q0 > qlim[1]):
        raise ValueError("RMRC starting joint positions are outside their limits")
    if not np.allclose(robot.fkine(q0).A, T1.A, atol=1e-4, rtol=0):
        raise RuntimeError("RMRC starting pose does not match the robots current joints")
    
    q_matrix[0, :] = q0

    for i in range(steps - 1):
        T_actual = robot.fkine(q_matrix[i, :])
        lin_vel = (traj[i + 1].t - T_actual.t) / delta_t
        # Relative rotation expressed in the world fram
        R_error = traj[i+1].R@T_actual.R.T
        # Rotation vector: Direction is rotation axis, magnitude is rotation angle in radians
        ang_vel = R.from_matrix(R_error).as_rotvec() / delta_t
        xdot = np.concatenate([lin_vel, ang_vel])

        J = robot.jacob0(q_matrix[i, :])
        # Damped least squares -- avoids qdot exploding near a singularity
        m = np.sqrt(max(np.linalg.det(J @ J.T), 0.0))
        damping = max_damping * (1 - m / min_manipulability) ** 2 if m < min_manipulability else 0.0

        A = J @ J.T + damping**2 * np.eye(6)
        # Solve the damped least squares system without explicitly inverting A
        q_dot = J.T@np.linalg.solve(A, xdot)

        # Conservative simulation speed limit: 90 degrees / second
        # Scale the whole vector to preserve its relative joint rate directions
        max_joint_speed = np.radians(90)
        speed_scale = max(1.0, np.max(np.abs(q_dot)) / max_joint_speed)
        q_dot = q_dot / speed_scale
        q_next = q_matrix[i, :] + q_dot * delta_t

        # Check next configuration is valid and within joint limits
        if not np.isfinite(q_next).all():
            raise RuntimeError(f"RMRC produced non-finite joint values at step {i+1}")
        if np.any(q_next < qlim[0]) or np.any(q_next > qlim[1]):
            raise RuntimeError(f"RMRC produced joint values outside their limits at step {i+1}")
        q_matrix[i + 1, :] = q_next

    # Checks that the final pose is within tolerance of the desired pose
    T_final = robot.fkine(q_matrix[-1, :])
    position_error = np.linalg.norm(T2.t - T_final.t)
    orientation_error = np.linalg.norm(R.from_matrix(T2.R@T_final.R.T).as_rotvec())

    if position_error > position_tolerance or orientation_error > orientation_tolerance:
        raise RuntimeError(f"RMRC failed to reach the final pose within tolerance: "
                            f"position error {position_error:.4f} m, "
                            f"orientation error {np.degrees(orientation_error):.2f} degrees")

    return q_matrix