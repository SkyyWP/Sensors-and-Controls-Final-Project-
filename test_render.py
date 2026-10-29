from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

root = Path(__file__).resolve().parent
model_path = root / "models" / "ur3e_model" / "ur3e.xml"
output_path = root / "test_render.png"

print("Loading model...", flush=True)

model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

print("Creating off-screen renderer...", flush=True)

with mujoco.Renderer(model, height=480, width=640) as renderer:
    renderer.update_scene(data)
    rgb = renderer.render()

Image.fromarray(np.asarray(rgb)).save(output_path)

print("RENDER SUCCESSFUL", flush=True)
print("Image dimensions:", rgb.shape, flush=True)
print("Saved to:", output_path, flush=True)
