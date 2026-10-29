from pathlib import Path
import sys
import time
import csv

import mujoco
import mujoco.viewer
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

XML = HERE / "models/ur3e_camera/scene_pick.xml"
CAMERA_NAME = "wrist_rgbd"
TARGET_JOINT_NAME = "target_free"

# Simulation / controller settings
DT = 0.002
CONTROL_EVERY = 10
DURATION = 12.0
TARGET_HEIGHT = 0.031
CAMERA_STANDOFF = 0.13
MAX_CARTESIAN_SPEED = 0.07
MAX_JOINT_SPEED = 0.45
DAMPING = 0.05
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480

model = mujoco.MjModel.from_xml_path(str(XML))
model.opt.timestep = DT
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

camera_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME
)
target_joint_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_JOINT, TARGET_JOINT_NAME
)

if camera_id < 0:
    raise RuntimeError(f"Camera '{CAMERA_NAME}' not found.")
if target_joint_id < 0:
    raise RuntimeError(f"Joint '{TARGET_JOINT_NAME}' not found.")

qadr = model.jnt_qposadr[:6].astype(int)
dadr = model.jnt_dofadr[:6].astype(int)
target_adr = int(model.jnt_qposadr[target_joint_id])
camera_body = int(model.cam_bodyid[camera_id])

# Randomise the red target close to the camera's starting field of view.
rng = np.random.default_rng()
mujoco.mj_forward(model, data)
cam_start = data.cam_xpos[camera_id].copy()
target_xy = cam_start[:2] + rng.uniform(-0.070, 0.070, size=2)

data.qpos[target_adr:target_adr + 3] = [
    target_xy[0], target_xy[1], TARGET_HEIGHT
]
data.qpos[target_adr + 3:target_adr + 7] = [1, 0, 0, 0]
mujoco.mj_forward(model, data)

# Camera intrinsics from the MuJoCo camera's vertical field of view.
fovy = np.deg2rad(float(model.cam_fovy[camera_id]))
fy = IMAGE_HEIGHT / (2.0 * np.tan(fovy / 2.0))
fx = fy
cx = (IMAGE_WIDTH - 1) / 2.0
cy = (IMAGE_HEIGHT - 1) / 2.0

renderer = mujoco.Renderer(
    model, height=IMAGE_HEIGHT, width=IMAGE_WIDTH
)

def detect_target():
    """Detect the red target and back-project its median depth to world coordinates."""
    renderer.update_scene(data, camera=CAMERA_NAME)
    rgb = renderer.render().copy()

    red = (
        (rgb[:, :, 0] > 80)
        & (rgb[:, :, 0] > 1.35 * rgb[:, :, 1])
        & (rgb[:, :, 0] > 1.35 * rgb[:, :, 2])
    )

    ys, xs = np.where(red)
    if len(xs) < 20:
        return None

    # Restrict depth sampling to detected red pixels.
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=CAMERA_NAME)
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()

    z = depth[ys, xs]
    good = np.isfinite(z) & (z > 0.02) & (z < 2.0)
    if np.count_nonzero(good) < 20:
        return None

    xs, ys, z = xs[good], ys[good], z[good]
    depth_m = float(np.median(z))
    u, v = float(np.median(xs)), float(np.median(ys))

    # MuJoCo camera coordinates: forward is -Z, image-down is +Y.
    p_camera = np.array([
        (u - cx) * depth_m / fx,
        -(v - cy) * depth_m / fy,
        -depth_m
    ])

    cam_pos = data.cam_xpos[camera_id].copy()
    cam_rot = data.cam_xmat[camera_id].reshape(3, 3).copy()
    p_world = cam_pos + cam_rot @ p_camera

    return u, v, p_world, cam_pos, len(xs)

def camera_jacobian():
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jac(
        model, data, jacp, jacr,
        data.cam_xpos[camera_id].copy(), camera_body
    )
    return jacp[:, dadr], jacr[:, dadr]

def rotation_error(R_current, R_desired):
    # Axis-angle error expressed in world coordinates.
    R_err = R_desired @ R_current.T
    return 0.5 * np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1],
    ])

print("\n========== LIVE RGB-D ROBOT DEMO ==========")
print("Model:", XML)
print("Camera:", CAMERA_NAME)
print("Random target XY:", np.round(target_xy, 3))
print("Controller: depth-based position visual servoing")
print("Close the MuJoCo viewer to stop.")
print("===========================================\n")

records = []
q_command = data.qpos[qadr].copy()
last_detection = None
start = time.perf_counter()
total_steps = int(DURATION / DT)

try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for step in range(total_steps):
            t = step * DT

            if not viewer.is_running():
                break

            if step % CONTROL_EVERY == 0:
                detection = detect_target()

                if detection is not None:
                    u, v, target_world, cam_pos, pixel_count = detection
                    last_detection = detection

                    # Keep the camera above the detected target surface.
                    desired_cam = target_world + np.array([
                        0.0, 0.0, CAMERA_STANDOFF
                    ])
                    cart_error = desired_cam - cam_pos

                    # Cartesian proportional control with speed limit.
                    v_cart = 1.5 * cart_error
                    speed = np.linalg.norm(v_cart)
                    if speed > MAX_CARTESIAN_SPEED:
                        v_cart *= MAX_CARTESIAN_SPEED / speed

                    Jp, Jr = camera_jacobian()

                    # Preserve the initial camera orientation, which points down.
                    if "R_desired" not in globals():
                        R_desired = data.cam_xmat[camera_id].reshape(3, 3).copy()

                    R_current = data.cam_xmat[camera_id].reshape(3, 3).copy()
                    w_cart = 2.0 * rotation_error(R_current, R_desired)

                    # Combined 6D task: translation and wrist orientation.
                    J6 = np.vstack((Jp, Jr))
                    task_velocity = np.concatenate((v_cart, w_cart))

                    J_dls = J6.T @ np.linalg.inv(
                        J6 @ J6.T + DAMPING**2 * np.eye(6)
                    )
                    qdot = J_dls @ task_velocity
                    qdot = np.clip(
                        qdot, -MAX_JOINT_SPEED, MAX_JOINT_SPEED
                    )

                    # Position servos receive short incremental joint targets.
                    q_now = data.qpos[qadr].copy()
                    q_command = q_now + qdot * DT * CONTROL_EVERY
                    if step % 100 == 0:
                        print(
                            "JOINT DEBUG | qdot:",
                            np.round(qdot, 3),
                            "| command:",
                            np.round(q_command, 3),
                            "| actual:",
                            np.round(data.qpos[qadr], 3),
                        )
                    q_command = np.clip(
                        q_command,
                        model.actuator_ctrlrange[:6, 0],
                        model.actuator_ctrlrange[:6, 1]
                    )

                    distance = float(np.linalg.norm(cart_error))
                    records.append([
                        t, u, v, *target_world, *cam_pos,
                        distance, pixel_count
                    ])

                    if step % 100 == 0:
                        print(
                            f"t={t:5.2f}s | pixel=({u:5.1f},{v:5.1f}) "
                            f"| camera-target error={distance:.3f} m "
                            f"| red pixels={pixel_count}"
                        )
                else:
                    # Hold the last safe joint target if vision temporarily fails.
                    q_command = np.clip(
                        q_command,
                        model.actuator_ctrlrange[:6, 0],
                        model.actuator_ctrlrange[:6, 1]
                    )

            data.ctrl[:6] = q_command
            mujoco.mj_step(model, data)
            viewer.sync()

            # Allow the viewer to remain visible at approximately real time.
            time.sleep(DT)

finally:
    renderer.close()

output = HERE / "visual_servo_results.csv"
with output.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "time_s", "pixel_u", "pixel_v",
        "target_x_m", "target_y_m", "target_z_m",
        "camera_x_m", "camera_y_m", "camera_z_m",
        "position_error_m", "red_pixel_count"
    ])
    writer.writerows(records)

print("\nDemo finished.")
print("Successful vision updates:", len(records))
print("Trajectory saved to:", output)
