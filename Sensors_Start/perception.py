import numpy as np
import mujoco
from PIL import Image
from pathlib import Path

# --------------------------------------------------
# 1. Load the saved RGB-D images
# --------------------------------------------------
rgb = np.array(Image.open("wrist_rgb.png").convert("RGB"))
depth = np.load("wrist_depth.npy")

height, width = depth.shape

# --------------------------------------------------
# 2. Detect the red target in the RGB image
# --------------------------------------------------
r = rgb[:, :, 0].astype(float)
g = rgb[:, :, 1].astype(float)
b = rgb[:, :, 2].astype(float)

red_mask = (r > 100) & (r > 1.5 * g) & (r > 1.5 * b)

if np.count_nonzero(red_mask) < 20:
    raise RuntimeError(
        "Target not detected. Check wrist_rgb.png and adjust the red threshold."
    )

rows, cols = np.where(red_mask)

# Estimate the target's image centre.
u = float(np.mean(cols))
v = float(np.mean(rows))

# Use the median depth of detected red pixels.
target_depth = float(np.median(depth[red_mask]))

# --------------------------------------------------
# 3. Load the MuJoCo model to obtain camera pose
# --------------------------------------------------
model_path = Path("models/ur3e_camera/scene_pick.xml")
model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)

if model.nkey > 0:
    mujoco.mj_resetDataKeyframe(model, data, 0)
else:
    mujoco.mj_resetData(model, data)

mujoco.mj_forward(model, data)

camera_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_rgbd"
)

if camera_id < 0:
    raise RuntimeError("Camera 'wrist_rgbd' was not found in the model.")

camera_position = data.cam_xpos[camera_id].copy()
camera_rotation = data.cam_xmat[camera_id].reshape(3, 3).copy()

# --------------------------------------------------
# 4. Calculate camera intrinsics
# --------------------------------------------------
fovy_degrees = float(model.cam_fovy[camera_id])
fovy_radians = np.deg2rad(fovy_degrees)

fy = height / (2.0 * np.tan(fovy_radians / 2.0))
fx = fy  # Square pixels
cx = (width - 1) / 2.0
cy = (height - 1) / 2.0

# --------------------------------------------------
# 5. Back-project the pixel into 3D camera coordinates
# --------------------------------------------------
# MuJoCo's camera looks along its local negative Z axis.
point_camera = np.array([
    (u - cx) * target_depth / fx,
    -(v - cy) * target_depth / fy,
    -target_depth
])

# Convert camera coordinates into world coordinates.
point_world = camera_position + camera_rotation @ point_camera

# --------------------------------------------------
# 6. Report the estimate
# --------------------------------------------------
print("\n========== RGB-D PERCEPTION RESULT ==========")
print(f"Image dimensions: {width} x {height}")
print(f"Detected red pixels: {len(cols)}")
print(f"Target image centre: u={u:.2f}, v={v:.2f} pixels")
print(f"Estimated target depth: {target_depth:.5f} m")
print(f"Camera vertical FOV: {fovy_degrees:.2f} degrees")
print(f"Camera focal length: fx={fx:.2f}, fy={fy:.2f} pixels")
print(f"Camera position (world): {camera_position}")
print(f"Target surface point (camera frame): {point_camera}")
print(f"Target surface point (world frame): {point_world}")
print("=============================================\n")

########### - Kalman Filter Integrataion - ###########

A = np.eye(3)
C = np.eye(3)


def camera_Rn(
    cam_rot,
    depth,
    fx,
    sigma_px=1.0,
    sigma_d0=0.002,
    sigma_dk=0.005,
):
    """
    RGB-D measurement covariance, expressed in world coordinates.

    cam_rot: camera optical-frame-to-world rotation (3x3)
    depth:   measured depth in metres
    fx:      horizontal focal length in pixels
    """
    cam_rot = np.asarray(cam_rot, dtype=float).reshape(3, 3)

    s_side = sigma_px * depth / fx
    s_depth = sigma_d0 + sigma_dk * depth**2

    Rn_cam = np.diag([
        s_side**2,
        s_side**2,
        s_depth**2,
    ])

    return cam_rot @ Rn_cam @ cam_rot.T


def kf_predict(xh, P, Qd):
    xh_pred = A @ xh
    P_pred = A @ P @ A.T + Qd
    return xh_pred, P_pred


def kf_update(xh_pred, P_pred, y, Rn):
    S = C @ P_pred @ C.T + Rn
    K = P_pred @ C.T @ np.linalg.inv(S)

    nu = y - C @ xh_pred
    xh = xh_pred + K @ nu

    P = (np.eye(3) - K @ C) @ P_pred
    P = 0.5 * (P + P.T)

    return xh, P, nu, S


def kf_step(xh, P, y, Qd, Rn):
    xh, P = kf_predict(xh, P, Qd)

    if y is None:
        return xh, P, None

    xh, P, nu, S = kf_update(xh, P, y, Rn)
    nis = float(nu @ np.linalg.solve(S, nu))

    return xh, P, nis


def _as_covariance(value):
    """Convert scalar, 3-vector or 3x3 matrix into a covariance matrix."""
    value = np.asarray(value, dtype=float)

    if value.ndim == 0:
        return np.eye(3) * float(value)

    if value.shape == (3,):
        return np.diag(value)

    if value.shape == (3, 3):
        return value.copy()

    raise ValueError(
        "Noise must be a scalar, 3-element vector or 3x3 matrix."
    )


class TargetKalmanFilter:
    """
    Wrapper around the 3D Kalman Filter functions.

    State: target position [x, y, z] in world coordinates.
    """

    def __init__(self, process_noise, measurement_noise):
        self.Q = _as_covariance(process_noise)
        self.R = _as_covariance(measurement_noise)

        self.position = None
        self.P = None
        self.last_nis = None

    @property
    def uncertainty(self):
        """Standard deviation of estimated position along each axis."""
        if self.P is None:
            return np.full(3, np.inf)

        return np.sqrt(np.maximum(np.diag(self.P), 0.0))

    def predict(self, dt):
        if self.position is None:
            return None

        # Convert process-noise rate into discrete-time covariance.
        Qd = self.Q * max(float(dt), 0.0)

        self.position, self.P = kf_predict(
            self.position,
            self.P,
            Qd,
        )

        return self.position.copy()

    def update(self, measurement):
        """
        Accepts either:
          1. A 3D world-position array.
          2. An RGB-D perception dictionary containing 'world'.

        If the dictionary also contains 'cam_rot', 'depth' and 'fx',
        camera_Rn() is used to calculate measurement covariance.
        Otherwise, the configured measurement_noise is used.
        """

        if isinstance(measurement, dict):
            if not measurement.get("detected", False):
                return self.position

            y = np.asarray(
                measurement["world"], dtype=float
            ).reshape(3)

            if all(
                key in measurement
                for key in ("cam_rot", "depth", "fx")
            ):
                Rn = camera_Rn(
                    measurement["cam_rot"],
                    measurement["depth"],
                    measurement["fx"],
                )
            else:
                Rn = self.R

        else:
            y = np.asarray(measurement, dtype=float).reshape(3)
            Rn = self.R

        if not np.all(np.isfinite(y)):
            raise ValueError("Invalid target position measurement.")

        # Initialise using the first valid camera measurement.
        if self.position is None:
            self.position = y.copy()
            self.P = Rn.copy()
            self.last_nis = None
            return self.position.copy()

        self.position, self.P, nu, S = kf_update(
            self.position,
            self.P,
            y,
            Rn,
        )

        self.last_nis = float(
            nu @ np.linalg.solve(S, nu)
        )

        return self.position.copy()
