"""Render an RGB and depth frame from the fixed wrist camera.

Run with: mjpython test_camera.py

The camera mount is not changed at runtime. For this diagnostic
only, the target is moved along the camera viewing ray so we can
verify that RGB and depth rendering work.
"""
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "models" / "ur3e_camera" / "scene_pick.xml"
WIDTH, HEIGHT = 640, 480

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

# Use the first keyframe if one exists; otherwise use model defaults.
if model.nkey > 0:
    mujoco.mj_resetDataKeyframe(model, data, 0)
else:
    mujoco.mj_resetData(model, data)

camera_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_rgbd"
)
target_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "target"
)

if camera_id < 0:
    raise RuntimeError("Camera 'wrist_rgbd' was not found.")
if target_id < 0:
    raise RuntimeError("Target body was not found.")

mujoco.mj_forward(model, data)

# Calculate the fixed camera's world-space viewing ray.
camera_position = data.cam_xpos[camera_id].copy()
camera_rotation = data.cam_xmat[camera_id].reshape(3, 3)
camera_forward = -camera_rotation[:, 2]
camera_forward /= np.linalg.norm(camera_forward)

# Diagnostic only: move the free target 0.45 m along the viewing ray.
# This changes simulation state in memory, not either XML file.
joint_id = int(model.body_jntadr[target_id])
if joint_id < 0:
    raise RuntimeError("Target body has no joint.")
if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
    raise RuntimeError("Target must have a free joint for this test.")

qadr = int(model.jnt_qposadr[joint_id])
test_target_position = camera_position + 0.45 * camera_forward
data.qpos[qadr:qadr + 3] = test_target_position
mujoco.mj_forward(model, data)

print("Camera position:", np.round(camera_position, 3))
print("Camera forward:", np.round(camera_forward, 3))
print("Diagnostic target position:", np.round(test_target_position, 3))

with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
    renderer.update_scene(data, camera="wrist_rgbd")
    rgb = renderer.render()

    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera="wrist_rgbd")
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()

Image.fromarray(rgb).save(HERE / "wrist_rgb.png")
np.save(HERE / "wrist_depth.npy", depth)

# Make a preview using foreground pixels, excluding far background.
far = float(model.stat.extent * model.vis.map.zfar)
valid = np.isfinite(depth) & (depth > 0) & (depth < 0.99 * far)

preview = np.zeros(depth.shape, dtype=np.uint8)
if np.any(valid):
    near, distant = np.percentile(depth[valid], [2, 98])
    if distant > near:
        scaled = np.clip((depth - near) / (distant - near), 0, 1)
        preview[valid] = (255 * (1 - scaled[valid])).astype(np.uint8)
    else:
        preview[valid] = 255

Image.fromarray(preview).save(HERE / "wrist_depth_preview.png")

print("RGB image:", HERE / "wrist_rgb.png")
print("Depth array:", HERE / "wrist_depth.npy")
print("Depth preview:", HERE / "wrist_depth_preview.png")
print("Image shapes:", rgb.shape, depth.shape)
print("Foreground pixels:", int(valid.sum()), "/", depth.size)

if np.any(valid):
    print(
        "Foreground depth range: %.4f to %.4f m"
        % (float(depth[valid].min()), float(depth[valid].max()))
    )
else:
    print("No foreground depth pixels detected; inspect camera pose and clipping.")
