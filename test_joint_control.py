"""Baseline test of joint observer and position feedback controller."""

from pathlib import Path
import sys

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from SRC.Estimation.Joint_observer import JointObserver
from SRC.Control.Joint_controller import JointController


XML = HERE / "models/ur3e_camera/scene_pick.xml"
OUTPUT = HERE / "joint_control_results.csv"

model = mujoco.MjModel.from_xml_path(str(XML))
data = mujoco.MjData(model)

mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

dt = model.opt.timestep
duration = 2.0
steps = int(duration / dt)

joint_ids = list(range(6))
q_addresses = model.jnt_qposadr[joint_ids].astype(int)
v_addresses = model.jnt_dofadr[joint_ids].astype(int)

q_initial = data.qpos[q_addresses].copy()
q_reference = q_initial.copy()
q_reference[0] += 0.15

observers = [
    JointObserver(dt=dt, measurement_std=0.001, acceleration_std=2.0)
    for _ in range(6)
]
controller = JointController(model)

rng = np.random.default_rng(7)
records = []

print("\n========== JOINT CONTROL TEST ==========")
print("Duration:", duration, "s")
print("Timestep:", dt, "s")
print("Reference change: shoulder pan +0.15 rad")
print("Position measurement noise std: 0.001 rad")

for step in range(steps):
    t = step * dt

    # Position-only measurements with simulated sensor noise.
    q_true = data.qpos[q_addresses].copy()
    q_measured = q_true + rng.normal(0.0, 0.001, size=6)

    estimates = np.array([
        observers[i].update(q_measured[i])
        for i in range(6)
    ])
    q_estimated = estimates[:, 0]
    qd_estimated = estimates[:, 1]

    data.ctrl[:6] = controller.command(
        q_reference, q_measured, qd_estimated
    )

    mujoco.mj_step(model, data)

    q_after = data.qpos[q_addresses].copy()
    qd_true_after = data.qvel[v_addresses].copy()

    records.append([
        t,
        *q_reference,
        *q_measured,
        *q_estimated,
        *qd_estimated,
        *q_after,
        *qd_true_after,
    ])

records = np.asarray(records)

headers = (
    ["time_s"]
    + [f"q_ref_{i+1}" for i in range(6)]
    + [f"q_measured_{i+1}" for i in range(6)]
    + [f"q_estimated_{i+1}" for i in range(6)]
    + [f"qd_estimated_{i+1}" for i in range(6)]
    + [f"q_true_{i+1}" for i in range(6)]
    + [f"qd_true_{i+1}" for i in range(6)]
)

np.savetxt(
    OUTPUT, records, delimiter=",",
    header=",".join(headers), comments=""
)

q_final = data.qpos[q_addresses].copy()
position_error = q_reference - q_final

print("\nFinal joint positions (rad):")
print(np.round(q_final, 4))
print("\nReference joint positions (rad):")
print(np.round(q_reference, 4))
print("\nFinal position errors (rad):")
print(np.round(position_error, 4))
print("\nFinal shoulder-pan error (deg):",
      round(float(np.rad2deg(position_error[0])), 3))
print("Results saved to:", OUTPUT)
print("========================================\n")
