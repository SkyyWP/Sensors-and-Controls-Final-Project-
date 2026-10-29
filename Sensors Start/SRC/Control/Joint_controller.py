"""Reference commands for MuJoCo's built-in position servos."""

import numpy as np


class JointController:
    def __init__(self, model):
        self.model = model
        self.lower = model.actuator_ctrlrange[:6, 0].copy()
        self.upper = model.actuator_ctrlrange[:6, 1].copy()

    def command(self, reference, measured_position, estimated_velocity):
        reference = np.asarray(reference, dtype=float)

        if reference.shape != (6,):
            raise ValueError("Reference must contain six joint positions.")

        # MuJoCo's position actuators already implement feedback and damping.
        # The observer estimates are retained for logging and evaluation.
        return np.clip(reference, self.lower, self.upper)
