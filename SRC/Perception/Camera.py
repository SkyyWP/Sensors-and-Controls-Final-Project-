"""Camera intrinsics and current MuJoCo camera pose."""
import numpy as np
import mujoco


def get_camera_id(model, camera_name="wrist_rgbd"):
    camera_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
    )
    if camera_id < 0:
        raise ValueError(f"Camera not found: {camera_name}")
    return camera_id


def get_camera_intrinsics(model, camera_id, width, height):
    fovy = np.deg2rad(float(model.cam_fovy[camera_id]))
    fy = height / (2.0 * np.tan(fovy / 2.0))
    fx = fy

    return {
        "fx": fx,
        "fy": fy,
        "cx": (width - 1) / 2.0,
        "cy": (height - 1) / 2.0,
        "width": width,
        "height": height,
    }


def get_camera_pose(model, data, camera_name="wrist_rgbd"):
    camera_id = get_camera_id(model, camera_name)

    # These values come from the current simulation state.
    position = data.cam_xpos[camera_id].copy()
    rotation = data.cam_xmat[camera_id].reshape(3, 3).copy()

    return position, rotation
