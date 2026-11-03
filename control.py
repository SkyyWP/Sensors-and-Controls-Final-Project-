import numpy as np
import mujoco

from config import IK_DAMPING, IK_MAX_ITERATIONS, IK_STEP_LIMIT


class JointLQRController:
    """
    Plug-in interface for the teammate's LQR controller.

    Until their gain matrix is supplied, the baseline uses the UR3e
    position actuators to track an IK-generated joint reference.
    """

    def __init__(self, K=None):
        self.K = K

    def command(self, q_reference, q_estimated, qd_estimated):
        if self.K is None:
            return np.asarray(q_reference, dtype=float)

        # Replace this mapping when the team's actuator-space LQR is ready.
        error = np.concatenate((
            q_estimated - q_reference,
            qd_estimated
        ))
        return np.asarray(q_reference) - self.K @ error


def solve_position_ik(model, data, site_id, target_position):
    """
    Damped-least-squares position IK for the six UR3e revolute joints.
    Orientation is not constrained in this first baseline.
    """

    joint_ids = list(range(6))
    qpos_addresses = [int(model.jnt_qposadr[j]) for j in joint_ids]
    dof_addresses = [int(model.jnt_dofadr[j]) for j in joint_ids]

    original_qpos = data.qpos.copy()
    q = np.array([data.qpos[a] for a in qpos_addresses])

    target = np.asarray(target_position, dtype=float)

    for _ in range(IK_MAX_ITERATIONS):
        for i, address in enumerate(qpos_addresses):
            data.qpos[address] = q[i]

        mujoco.mj_forward(model, data)

        current = data.site_xpos[site_id].copy()
        error = target - current

        if np.linalg.norm(error) < 0.003:
            break

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

        J = jacp[:, dof_addresses]
        delta = J.T @ np.linalg.solve(
            J @ J.T + (IK_DAMPING ** 2) * np.eye(3),
            error
        )

        delta = np.clip(delta, -IK_STEP_LIMIT, IK_STEP_LIMIT)
        q += delta

        # Respect joint limits.
        for i, joint_id in enumerate(joint_ids):
            if model.jnt_limited[joint_id]:
                low, high = model.jnt_range[joint_id]
                q[i] = np.clip(q[i], low, high)

    data.qpos[:] = original_qpos
    mujoco.mj_forward(model, data)

    return q


def read_arm_positions(model, data):
    return np.array([
        data.qpos[int(model.jnt_qposadr[j])]
        for j in range(6)
    ])
