"""
Estimation: Kalman filter for the target position and observer for the joint velocities.
 
Usage:
    target_filter = TargetKalmanFilter(process_noise, measurement_noise)
    target_filter.predict(dt)                  every camera frame
    target_filter.update(measurement)          when the target is detected
    target_filter.position                     target position [x, y, z] in the world frame [m]
 
    observer = JointStateObserver()
    q_est, qd_est = observer.update(q, dt)     every control step (dt = physics timestep)
"""

import numpy as np
from kalman import kf_predict, kf_update, camera_Rn


class TargetKalmanFilter:
    def __init__(self, process_noise, measurement_noise, initial_sigma=0.1):
        # target drift, 1-sigma [m per sqrt(s)]
        self.q = float(process_noise)
        # default measurement 1-sigma [m]
        self.r = float(measurement_noise)
        self.initial_sigma = float(initial_sigma)
        self.reset()

    def reset(self):
        self.position = None
        self.P = np.diag([self.initial_sigma**2] * 3)
        self.nis = []

    def predict(self, dt):
        if self.position is None:
            return
        Qd = np.diag([self.q**2 * dt] * 3)
        self.position, self.P = kf_predict(self.position, self.P, Qd)

    def update(self, measurement, depth=None, cam_rot=None, fx=None):
        """
        Correct the estimate with one RGB-D measurement.

        If depth, cam_rot and fx are given, Rn depends on the camera pose (camera_Rn).
        Otherwise Rn uses measurement_noise in every direction.

        Returns:
            Updated target position estimate [x, y, z] in the world frame [m].
        """
        y = np.asarray(measurement, dtype=float).reshape(3)

        if depth is not None and cam_rot is not None and fx is not None:
            Rn = camera_Rn(np.asarray(cam_rot).reshape(3, 3), depth, fx)
        else:
            Rn = np.diag([self.r**2] * 3)

 # Redone:
        # if self.position is None:
            # self.position = y.copy()            # first detection initialises the filter
            # return self.position

        if self.position is None:
            self.position = y.copy()
            self.P = Rn.copy()
            return self.position.copy()

        self.position, self.P, nu, S = kf_update(self.position, self.P, y, Rn)
        self.nis.append(float(nu @ np.linalg.solve(S, nu)))
        return self.position

    @property
    def uncertainty(self):
        return np.sqrt(np.diag(self.P))         # 1-sigma in x, y, z [m]


class JointStateObserver:
    def __init__(self, n_joints=6, q_angle=1e-8, q_vel=1e-4, angle_noise=1e-3):
        self.n = n_joints
        # process noise variance on the angle, per step [rad^2]
        self.q_angle = float(q_angle)
        # process noise variance on the velocity, per step [(rad/s)^2]
        self.q_vel = float(q_vel)
        # joint angle reading, 1-sigma [rad]
        self.sr = float(angle_noise)
        self.reset()

    def reset(self):
        # (n, 2): [q, qd] for each joint
        self.x = None
        self.P = np.array([np.diag([self.sr**2, 1.0]) for _ in range(self.n)])

    def update(self, q_measured, dt):
        """
        One predict + correct step for every joint. Only the angle is measured,
        the velocity is inferred from how the angle changes.

        Returns:
            q_est     Estimated joint angles (n,) [rad]
            qd_est    Estimated joint velocities (n,) [rad/s]
        """
        q_measured = np.asarray(q_measured, dtype=float).reshape(self.n)

        if self.x is None:
            self.x = np.column_stack([q_measured, np.zeros(self.n)])
            return self.x[:, 0].copy(), self.x[:, 1].copy()

        # constant velocity model, same as the LQR
        A = np.array([[1.0, dt], [0.0, 1.0]])
        C = np.array([[1.0, 0.0]])              # only the angle is measured
        # tuned for the 2 ms physics step: retune if dt changes
        Qd = np.diag([self.q_angle, self.q_vel])
        Rn = np.array([[self.sr**2]])

        for j in range(self.n):
            xh = A @ self.x[j]
            P = A @ self.P[j] @ A.T + Qd
            S = C @ P @ C.T + Rn
            K = P @ C.T @ np.linalg.inv(S)
            xh = xh + K @ (np.array([q_measured[j]]) - C @ xh)
            P = (np.eye(2) - K @ C) @ P
            self.x[j], self.P[j] = xh, 0.5 * (P + P.T)

        return self.x[:, 0].copy(), self.x[:, 1].copy()
