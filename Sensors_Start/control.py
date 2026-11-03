import numpy as np
import mujoco
from scipy.linalg import solve_discrete_are #Requires SCI PY to compute P aka Riccata equation
from config import IK_DAMPING, IK_MAX_ITERATIONS, IK_STEP_LIMIT


class JointLQRController:
    """
    Plug-in interface for the teammate's LQR controller.

    Until their gain matrix is supplied, the baseline uses the UR3e
    position actuators to track an IK-generated joint reference.
    """

    def __init__(self, dt):
        self.dt = dt
        
        #LQR Setup
        A = np.array([[1, dt], [0, 1]]) #caculates next angle 
        B = np.array([[dt**2/2], [dt]]) # caculates the next speed

        controllable =np.hstack((B, A @ B))
        if np.linalg.matrix_rank(controllable) == 2:
            print("System is controllable")

        Q = np.diag([1/0.1**2, 1]) #Error Cost
        R = np.array([[1/0.1**2]]) #Effort Cost

        P = solve_discrete_are(A, B, Q, R) #Solves Riccati Equation
        
        self.K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A) #Computes the gain

    def command(self, q_goal, q_est, qd_est, qd_ref=None, qdd_ref=None):
        
        qd_ref = np.zeros(6) if qd_ref is None else qd_ref
        qdd_ref = np.zeros(6) if qdd_ref is None else qdd_ref
        dt = self.dt
       
        error_position = q_est - q_goal #gets distance from the target from each joint
        error_velocity = qd_est - qd_ref # gets current speed
        u = -(self.K[0, 0] * error_position + self.K[0, 1] * error_velocity) + qdd_ref
        
        #returns q next which is = estimated q + q*dt + 1/2 * u * dt^2
        return q_est + qd_est * dt + 0.5 * u * dt**2


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
