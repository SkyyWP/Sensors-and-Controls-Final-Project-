import numpy as np
import mujoco
from mujoco import Renderer

from config import (
    CAMERA_NAME, IMAGE_WIDTH, IMAGE_HEIGHT,
    MIN_RED_PIXELS, RED_THRESHOLD, RED_RATIO,
    SURFACE_TO_CENTRE
)


class RGBDPerception:
    """Live wrist-camera RGB-D capture and 3D target localisation."""

    def __init__(self, model):
        self.model = model
        self.camera_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME
        )

        if self.camera_id < 0:
            raise RuntimeError(f"Camera '{CAMERA_NAME}' not found.")

        self.renderer = Renderer(
            model, height=IMAGE_HEIGHT, width=IMAGE_WIDTH
        )

        fovy = np.deg2rad(float(model.cam_fovy[self.camera_id]))
        self.fy = IMAGE_HEIGHT / (2 * np.tan(fovy / 2))
        self.fx = self.fy
        self.cx = (IMAGE_WIDTH - 1) / 2
        self.cy = (IMAGE_HEIGHT - 1) / 2

    def close(self):
        self.renderer.close()

    def capture(self, data):
        """Return RGB, depth, and estimated target world position."""

        self.renderer.update_scene(data, camera=CAMERA_NAME)
        rgb = self.renderer.render().copy()

        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(data, camera=CAMERA_NAME)
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()

        r = rgb[:, :, 0].astype(float)
        g = rgb[:, :, 1].astype(float)
        b = rgb[:, :, 2].astype(float)

        mask = (
            (r > RED_THRESHOLD)
            & (r > RED_RATIO * g)
            & (r > RED_RATIO * b)
        )

        rows, cols = np.where(mask)

        if len(cols) < MIN_RED_PIXELS:
            return {
                "detected": False,
                "rgb": rgb,
                "depth_image": depth,
                "pixel": None,
                "world": None,
                "count": len(cols),
            }

        # Use the median pixel and median depth of the red region.
        u = float(np.median(cols))
        v = float(np.median(rows))

        red_depth = depth[mask]
        valid = np.isfinite(red_depth) & (red_depth > 0.05) & (red_depth < 2.0)

        if np.count_nonzero(valid) < MIN_RED_PIXELS:
            return {
                "detected": False,
                "rgb": rgb,
                "depth_image": depth,
                "pixel": (u, v),
                "world": None,
                "count": len(cols),
            }

        d = float(np.median(red_depth[valid]))

        # MuJoCo camera frame: +X right, +Y up, -Z forward.
        point_camera = np.array([
            (u - self.cx) * d / self.fx,
            -(v - self.cy) * d / self.fy,
            -d,
        ])

        camera_position = data.cam_xpos[self.camera_id].copy()
        camera_rotation = data.cam_xmat[self.camera_id].reshape(3, 3).copy()

        point_surface_world = (
            camera_position + camera_rotation @ point_camera
        )

        camera_forward = -camera_rotation[:, 2]
        camera_forward /= np.linalg.norm(camera_forward)

        # Approximate the cube centre from its visible surface.
        point_world = (
            point_surface_world
            + SURFACE_TO_CENTRE * camera_forward
        )

        return {
            "detected": True,
            "rgb": rgb,
            "depth_image": depth,
            "pixel": (u, v),
            "depth_m": d,
            "world": point_world,
            "count": len(cols),

            # Kalman Filter measurement information:
            "fx": float(self.fx),
            "cam_rot": camera_rotation,
        }
