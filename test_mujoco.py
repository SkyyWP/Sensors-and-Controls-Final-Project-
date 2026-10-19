from pathlib import Path

import mujoco
import mujoco.viewer

# Locate the robot model relative to this Python file
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "ur3e_model" / "ur3e.xml"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"UR3e model not found: {MODEL_PATH}"
        )

    # Load the MJCF model and initialise simulation data
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)

    print("MuJoCo model loaded successfully.")
    print(f"Number of joints: {model.njnt}")
    print(f"Number of actuators: {model.nu}")
    print(f"Number of position coordinates: {model.nq}")
    print(f"Number of velocity coordinates: {model.nv}")

    print("\nJoints in model:")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_JOINT, i
        )
        print(f"  {i}: {name}")

    # Open the interactive MuJoCo simulation
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
