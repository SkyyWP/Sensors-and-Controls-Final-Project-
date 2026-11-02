"""
Target position estimator: Gets rgbd perception and adds kalman filter.
 
Usage:
    est = TargetEstimator(model)    once, at startup
    est.reset()                     at the start of each trial
    est.step(data)                  once per camera frame (about every 17 physics steps)
    est.xh                          target position [x, y, z] in the world frame [m] (note this it for the top of the apple not the centre)
"""

import numpy as np
import mujoco
from rgbd_perception import calculate_intrinsics, estimate_target_position
from kalman import kf_step, camera_Rn

FLIP = np.diag([1.0, -1.0, -1.0, 1.0])   # MuJoCo camera (looks -z, y up) -> optical frame (looks +z, y down)

class TargetEstimator:
    def __init__(self, model, cam_name="wrist_rgbd", width=640, height=480,
                 x0=None, sigma0=0.1, sigma_q=1e-6):
        self.model = model
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        self.cam_name = cam_name
        self.K = calculate_intrinsics(width, height, model.cam_fovy[self.cam_id])
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.Qd = np.diag([sigma_q**2] * 3)
        self.sigma0 = sigma0
        self.reset(x0)

    def reset(self, x0=None):
        self.xh = None if x0 is None else np.asarray(x0, dtype=float)
        self.P = np.diag([self.sigma0**2] * 3)
        self.nis = []

    def camera_pose(self, data):
        T = np.eye(4)
        T[:3, :3] = data.cam_xmat[self.cam_id].reshape(3, 3)
        T[:3, 3] = data.cam_xpos[self.cam_id]
        return T @ FLIP

    def render(self, data):
        self.renderer.update_scene(data, camera=self.cam_name)
        rgb = self.renderer.render()
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(data, camera=self.cam_name)
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        return rgb, depth

    def step(self, data):
        """
        Run one camera frame: render, detect the target, and update the filter.

        Returns:
            Current target position estimate [x, y, z] in the world frame [m].
 
            Returns None if the target has not been detected yet.
            If the target is not detected this frame, the filter predicts only.
        """
        rgb, depth = self.render(data)
        T = self.camera_pose(data)
        meas = estimate_target_position(rgb, depth, T, self.K)

        if meas is None:
            y, Rn = None, None
        else:
            y = meas["target_world"]
            Rn = camera_Rn(T[:3, :3], meas["depth_m"], self.K["fx"])

        if self.xh is None:
            if y is None:
                return None
            self.xh = y.copy()          # first detection initialises the filter
            return self.xh

        self.xh, self.P, nis = kf_step(self.xh, self.P, y, self.Qd, Rn)
        if nis is not None:
            self.nis.append(nis)
        return self.xh

    def sigma(self):
        return np.sqrt(np.diag(self.P))

    def converged(self, tol=0.002):
        return self.xh is not None and np.all(self.sigma() < tol)

    def close(self):
        self.renderer.close()
