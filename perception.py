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
