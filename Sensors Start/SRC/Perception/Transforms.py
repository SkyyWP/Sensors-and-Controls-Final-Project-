"""Pixel-to-3D and camera-to-world coordinate transformations."""
import numpy as np


def back_project_pixel(u, v, depth, intrinsics):
    """Convert a pixel and optical-axis depth to MuJoCo camera coordinates."""
    if not np.isfinite(depth) or depth <= 0:
        raise ValueError(f"Invalid depth: {depth}")

    x = (u - intrinsics["cx"]) * depth / intrinsics["fx"]
    y = -(v - intrinsics["cy"]) * depth / intrinsics["fy"]

    # MuJoCo cameras look along their local negative Z axis.
    return np.array([x, y, -depth], dtype=float)


def camera_to_world(point_camera, camera_position, camera_rotation):
    """Transform a point from MuJoCo camera coordinates to world coordinates."""
    return camera_position + camera_rotation @ point_camera
