"""
Camera noise: makes the perfect simulated RGB-D camera behave like a real one.
Each detection gets:
    - pixel noise on the detected centre (segmentation jitter), 1-sigma SIGMA_PX [px]
    - depth noise growing with distance, 1-sigma SIGMA_D0 + SIGMA_DK * depth^2 [m]
The noisy pixel and depth are back-projected the same way rgbd_perception.py does it.
"""

import numpy as np
from rgbd_perception import RGBDPerception
from config import SURFACE_TO_CENTRE

SIGMA_PX = 0.5          # centre-pixel noise, 1-sigma [px]
SIGMA_D0 = 0.001        # depth noise at zero range, 1-sigma [m]
SIGMA_DK = 0.01         # depth noise growth with range [1/m]: 0.01 * d^2


class NoisyRGBDPerception(RGBDPerception):
    def __init__(self, model, sigma_px=SIGMA_PX, sigma_d0=SIGMA_D0, sigma_dk=SIGMA_DK, seed=None):
        super().__init__(model)
        self.sigma_px = sigma_px
        self.sigma_d0 = sigma_d0
        self.sigma_dk = sigma_dk
        self.rng = np.random.default_rng(seed)

    def capture(self, data):
        result = super().capture(data)
        if not result["detected"]:
            return result
        return self.add_noise(result, data)

    def add_noise(self, result, data):
        """Perturb the detected pixel and depth, then back-project again."""
        u, v = result["pixel"]
        d = result["depth_m"]

        u = u + self.rng.normal(0.0, self.sigma_px)
        v = v + self.rng.normal(0.0, self.sigma_px)
        d = d + self.rng.normal(0.0, self.sigma_d0 + self.sigma_dk * d**2)

        # MuJoCo camera frame: +X right, +Y up, -Z forward (same as rgbd_perception.py)
        point_camera = np.array([
            (u - self.cx) * d / self.fx,
            -(v - self.cy) * d / self.fy,
            -d,
        ])
        camera_position = data.cam_xpos[self.camera_id]
        camera_rotation = data.cam_xmat[self.camera_id].reshape(3, 3)
        camera_forward = -camera_rotation[:, 2]

        noisy = dict(result)
        noisy["pixel"] = (u, v)
        noisy["depth_m"] = d
        noisy["world"] = camera_position + camera_rotation @ point_camera + SURFACE_TO_CENTRE * camera_forward
        return noisy
