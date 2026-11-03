from pathlib import Path
import xml.etree.ElementTree as ET

model_path = Path("models/ur3e_camera/ur3e.xml")

tree = ET.parse(model_path)
root = tree.getroot()

wrist = root.find(".//body[@name='wrist_3_link']")
if wrist is None:
    raise RuntimeError("Could not find wrist_3_link in the model.")

if root.find(".//camera[@name='wrist_rgbd']") is not None:
    print("Camera already exists; no changes made.")
else:
    ET.SubElement(
        wrist,
        "camera",
        {
            "name": "wrist_rgbd",
            "pos": "0 0 0.05",
            "quat": "1 0 0 0",
            "fovy": "60",
        },
    )

    tree.write(model_path, encoding="utf-8", xml_declaration=True)
    print(f"Added wrist_rgbd camera to {model_path}")

# Confirm MuJoCo can load the edited model.
import mujoco

model = mujoco.MjModel.from_xml_path(str(model_path))
camera_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_rgbd"
)

if camera_id < 0:
    raise RuntimeError("MuJoCo could not find the new camera.")

print("Camera-enabled model loaded successfully.")
print("Camera ID:", camera_id)
print("Number of cameras:", model.ncam)
