from pathlib import Path
import argparse
import time
import sys

import mujoco
import mujoco.viewer


HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "models" / "ur3e_camera" / "scene_pick.xml"


def load_simulation():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Could not find MuJoCo scene:\n{MODEL_PATH}"
        )

    print("\n======================================")
    print("       UR3e MuJoCo Simulation")
    print("======================================")
    print(f"Loading: {MODEL_PATH}")

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        print("Loaded initial keyframe.")
    else:
        mujoco.mj_resetData(model)
        print("Loaded default model configuration.")

    mujoco.mj_forward(model, data)

    print("\nMODEL INFORMATION")
    print(f"Joints:       {model.njnt}")
    print(f"Actuators:    {model.nu}")
    print(f"Bodies:       {model.nbody}")
    print(f"Simulation dt: {model.opt.timestep:.5f} s")

    # Hold position actuators at their initial joint positions.
    # This avoids commanding every actuator to zero at startup.
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])

        if joint_id < 0 or joint_id >= model.njnt:
            continue

        joint_type = model.jnt_type[joint_id]

        if joint_type in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            qpos_address = int(model.jnt_qposadr[joint_id])
            data.ctrl[actuator_id] = data.qpos[qpos_address]

    mujoco.mj_forward(model, data)

    return model, data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wrist-camera",
        action="store_true",
        help="Open the viewer using the wrist RGB-D camera."
    )
    args = parser.parse_args()

    model, data = load_simulation()

    camera_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_CAMERA,
        "wrist_rgbd"
    )

    if args.wrist_camera and camera_id < 0:
        raise RuntimeError(
            "The wrist_rgbd camera does not exist in this model."
        )

    print("\nOpening interactive MuJoCo viewer...")
    print("Close the viewer window or press Ctrl+C to stop.")
    print("Use the mouse to rotate, zoom and move around the scene.")

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:

            if args.wrist_camera:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = camera_id
                print("Wrist RGB-D camera view enabled.")

            print("\nSIMULATION RUNNING\n")

            while viewer.is_running():
                step_start = time.perf_counter()

                mujoco.mj_step(model, data)

                viewer.sync()

                elapsed = time.perf_counter() - step_start
                remaining = model.opt.timestep - elapsed

                if remaining > 0:
                    time.sleep(remaining)

    except KeyboardInterrupt:
        print("\nSimulation stopped.")

    print("MuJoCo session closed.")


if __name__ == "__main__":
    main()
