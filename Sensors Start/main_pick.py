from pathlib import Path
import argparse
import time
import numpy as np
import mujoco
import mujoco.viewer

from config import *
from rgbd_perception import RGBDPerception
from estimation import TargetKalmanFilter, JointStateObserver
from control import JointLQRController, solve_position_ik, read_arm_positions
from grasp import KinematicGrasp


def get_id(model, object_type, name):
    result = mujoco.mj_name2id(model, object_type, name)
    if result < 0:
        raise RuntimeError(f"Could not find model element: {name}")
    return result


def set_joint_targets(data, q):
    # This model has six position actuators mapped in joint order.
    data.ctrl[:6] = q[:6]


def step_sim(model, data, viewer, steps, q_reference=None, grasp=None):
    for i in range(steps):
        if not viewer.is_running():
            return False

        if q_reference is not None:
            set_joint_targets(data, q_reference)

        mujoco.mj_step(model, data)

        if grasp is not None:
            grasp.update()

        if i % 10 == 0:
            viewer.sync()

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--random-target",
        action="store_true",
        help="Randomise the block on the tabletop before starting."
    )
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model)

    mujoco.mj_forward(model, data)

    camera_id = get_id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    site_id = get_id(model, mujoco.mjtObj.mjOBJ_SITE, END_EFFECTOR_SITE)
    target_body_id = get_id(model, mujoco.mjtObj.mjOBJ_BODY, TARGET_BODY)
    target_joint_id = get_id(model, mujoco.mjtObj.mjOBJ_JOINT, TARGET_JOINT)

    if model.jnt_type[target_joint_id] != mujoco.mjtJoint.mjJNT_FREE:
        raise RuntimeError("Target must have a free joint.")

    target_qadr = int(model.jnt_qposadr[target_joint_id])

    if args.random_target:
        table_id = get_id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
        target_geom_id = get_id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "target_geom"
        )

        rng = np.random.default_rng(RANDOM_SEED)
        table_pos = model.geom_pos[table_id]
        table_size = model.geom_size[table_id]
        target_half_height = model.geom_size[target_geom_id, 2]

        # Conservative random region within the tabletop.
        x = table_pos[0] + rng.uniform(-0.55, 0.55) * table_size[0]
        y = table_pos[1] + rng.uniform(-0.55, 0.55) * table_size[1]
        z = table_pos[2] + table_size[2] + target_half_height

        data.qpos[target_qadr:target_qadr + 3] = [x, y, z]
        data.qpos[target_qadr + 3:target_qadr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(model, data)

        print("Randomised target position:", np.round([x, y, z], 3))

    q_home = read_arm_positions(model, data)

    perception = RGBDPerception(model)
    target_filter = TargetKalmanFilter(
        TARGET_PROCESS_NOISE, TARGET_MEASUREMENT_NOISE
    )
    observer = JointStateObserver()
    controller = JointLQRController()

    grasp = KinematicGrasp(
        model, data, target_joint_id, site_id
    )

    print("\n======================================")
    print("UR3e RGB-D INTEGRATED PICK PIPELINE")
    print("======================================")
    print("Camera:", CAMERA_NAME)
    print("End effector:", END_EFFECTOR_SITE)
    print("Target:", TARGET_BODY)
    print("Mode: RGB-D PBVS baseline")
    print("Starting interactive simulation...\n")

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:

            # Keep the robot at its starting configuration.
            set_joint_targets(data, q_home)

            # -------------------------------------------------
            # PHASE 1: SCAN
            # -------------------------------------------------
            print("[1/6] Scanning for the red target...")

            scan_poses = [
                q_home.copy(),
                q_home + np.array([0.25, 0, 0, 0, 0, 0]),
                q_home + np.array([-0.25, 0, 0, 0, 0, 0]),
                q_home + np.array([0, 0.15, 0, 0, 0, 0]),
                q_home + np.array([0, -0.15, 0, 0, 0, 0]),
            ]

            measurement = None

            for scan_index, scan_q in enumerate(scan_poses):
                scan_q = np.clip(scan_q, -3.0, 3.0)

                if not step_sim(
                    model, data, viewer, SCAN_STEPS, scan_q
                ):
                    return

                mujoco.mj_forward(model, data)
                result = perception.capture(data)

                print(
                    f"Scan {scan_index + 1}: "
                    f"detected={result['detected']}, "
                    f"red pixels={result['count']}"
                )

                if result["detected"]:
                    measurement = result["world"]
                    break

            if measurement is None:
                print(
                    "\nTarget not found during scan. "
                    "Check camera orientation, target visibility and "
                    "red segmentation threshold."
                )
                return

            estimate = target_filter.update(measurement)
            print("Initial RGB-D target estimate:", np.round(estimate, 4))

            # -------------------------------------------------
            # PHASE 2: APPROACH
            # -------------------------------------------------
            print("\n[2/6] Planning approach using RGB-D position...")

            last_capture_time = data.time
            last_seen = data.time
            q_goal = None
            phase = "approach"

            for step in range(MAX_APPROACH_STEPS):
                if not viewer.is_running():
                    return

                dt = max(model.opt.timestep, data.time - last_capture_time)

                if step % CAPTURE_INTERVAL == 0:
                    result = perception.capture(data)

                    if result["detected"]:
                        target_filter.predict(dt)
                        estimate = target_filter.update(result["world"])
                        last_seen = data.time

                        print(
                            f"Target update | pixel={result['pixel']} "
                            f"| position={np.round(estimate, 3)} "
                            f"| uncertainty={np.round(target_filter.uncertainty, 4)}"
                        )
                    else:
                        target_filter.predict(dt)
                        estimate = target_filter.position

                    last_capture_time = data.time

                    if data.time - last_seen > 3.0:
                        print("Target lost for 3 seconds. Stopping safely.")
                        return

                    if phase == "approach":
                        desired = estimate + np.array(
                            [0, 0, APPROACH_HEIGHT]
                        )
                    else:
                        desired = estimate + np.array(
                            [0, 0, GRASP_HEIGHT]
                        )

                    q_goal = solve_position_ik(
                        model, data, site_id, desired
                    )

                q_actual = read_arm_positions(model, data)
                q_est, qd_est = observer.update(
                    q_actual, model.opt.timestep * CAPTURE_INTERVAL
                )

                q_command = controller.command(
                    q_goal, q_est, qd_est
                )
                set_joint_targets(data, q_command)

                mujoco.mj_step(model, data)

                if step % 10 == 0:
                    viewer.sync()

                site_position = data.site_xpos[site_id].copy()
                desired_site = (
                    estimate + np.array([0, 0, APPROACH_HEIGHT])
                    if phase == "approach"
                    else estimate + np.array([0, 0, GRASP_HEIGHT])
                )

                cartesian_error = np.linalg.norm(
                    desired_site - site_position
                )

                if step % 500 == 0:
                    print(
                        f"{phase}: cartesian error={cartesian_error:.4f} m"
                    )

                if phase == "approach" and cartesian_error < 0.025:
                    print("\n[3/6] Approach pose reached.")
                    phase = "descend"
                    q_goal = None

                elif phase == "descend" and cartesian_error < 0.018:
                    print("\n[4/6] Grasp position reached.")
                    break

            else:
                print("Approach timed out; no grasp attempted.")
                return

            # -------------------------------------------------
            # PHASE 3: GRASP
            # -------------------------------------------------
            print("\n[5/6] Activating simulated grasp...")
            grasp.attach()

            # -------------------------------------------------
            # PHASE 4: LIFT
            # -------------------------------------------------
            print("[6/6] Lifting target...")

            current_site = data.site_xpos[site_id].copy()
            lift_target = current_site + np.array(
                [0, 0, LIFT_HEIGHT]
            )

            q_lift = solve_position_ik(
                model, data, site_id, lift_target
            )

            step_sim(
                model, data, viewer, 1800,
                q_lift, grasp
            )

            print("\nPICK SEQUENCE FINISHED.")
            print("Note: grasp is currently a kinematic placeholder.")
            print("Close the viewer to end the simulation.")

            while viewer.is_running():
                mujoco.mj_step(model, data)
                grasp.update()
                viewer.sync()

    finally:
        perception.close()


if __name__ == "__main__":
    main()
