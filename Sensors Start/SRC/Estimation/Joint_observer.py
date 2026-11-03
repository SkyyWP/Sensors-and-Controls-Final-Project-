import numpy as np


class JointObserver:
    def __init__(self, dt, measurement_std=0.001, acceleration_std=2.0):
        self.dt = float(dt)
        self.A = np.array([
            [1.0, self.dt],
            [0.0, 1.0],
        ])
        self.H = np.array([[1.0, 0.0]])

        dt = self.dt
        sa2 = float(acceleration_std) ** 2
        self.Q = sa2 * np.array([
            [dt**4 / 4.0, dt**3 / 2.0],
            [dt**3 / 2.0, dt**2],
        ])
        self.R = np.array([[float(measurement_std) ** 2]])

        self.x = np.zeros(2)
        self.P = np.diag([1e-4, 1.0])
        self.initialised = False

    def update(self, measured_position):
        z = float(measured_position)

        if not self.initialised:
            self.x = np.array([z, 0.0])
            self.initialised = True
            return self.x.copy()

        # Predict
        x_pred = self.A @ self.x
        P_pred = self.A @ self.P @ self.A.T + self.Q

        # Correct using position measurement only
        innovation = z - float((self.H @ x_pred)[0])
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        self.x = x_pred + (K[:, 0] * innovation)

        # Joseph form helps preserve covariance symmetry and positivity
        I = np.eye(2)
        KH = K @ self.H
        self.P = (
            (I - KH) @ P_pred @ (I - KH).T
            + K @ self.R @ K.T
        )
        self.P = 0.5 * (self.P + self.P.T)

        return self.x.copy()

    @property
    def position(self):
        return float(self.x[0])

    @property
    def velocity(self):
        return float(self.x[1])
