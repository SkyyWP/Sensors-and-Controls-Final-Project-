"""Red-target detection and RGB-D 3D localisation."""
import numpy as np

from .Camera import get_camera_id, get_camera_intrinsics, get_camera_pose
from .Transforms import back_project_pixel, camera_to_world


def detect_red_target(rgb):
    """Return a red-pixel mask and its centroid."""
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("Expected an RGB image with three colour channels.")

    r = rgb[:, :, 0].astype(float)
    g = rgb[:, :, 1].astype(float)
    b = rgb[:, :, 2].astype(float)

    mask = (r > 100) & (r > 1.5 * g) & (r > 1.5 * b)

    rows, cols = np.where(mask)
    if len(cols) < 20:
        raise RuntimeError(
            "Target not detected. Check the RGB image and red threshold."
        )

    return mask, float(np.mean(cols)), float(np.mean(rows))


def estimate_target(rgb, depth, model, data, camera_name="wrist_rgbd"):
    """Estimate the detected target surface point in world coordinates.

    The model and data must be the same simulation state that generated
    the supplied RGB and depth images.
    """
    if depth.ndim != 2 or rgb.shape[:2] != depth.shape:
        raise ValueError("RGB and depth image dimensions do not match.")

    mask, u, v = detect_red_target(rgb)

    valid_depth = mask & np.isfinite(depth) & (depth > 0)
    if not np.any(valid_depth):
        raise RuntimeError("No valid depth measurements for the target.")

    target_depth = float(np.median(depth[valid_depth]))

    height, width = depth.shape
    camera_id = get_camera_id(model, camera_name)
    intrinsics = get_camera_intrinsics(model, camera_id, width, height)

    camera_position, camera_rotation = get_camera_pose(
        model, data, camera_name
    )

    point_camera = back_project_pixel(u, v, target_depth, intrinsics)
    point_world = camera_to_world(
        point_camera, camera_position, camera_rotation
    )

    return {
        "pixel": np.array([u, v]),
        "depth": target_depth,
        "pixel_count": int(mask.sum()),
        "intrinsics": intrinsics,
        "camera_position": camera_position,
        "camera_rotation": camera_rotation,
        "point_camera": point_camera,
        "point_world": point_world,
    }
