from pathlib import Path
import sys

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from SRC.Estimation.Joint_observer import JointObserver

XML = HERE / "models/ur3e_camera/scene_pick.xml"
OUTPUT = HERE / "observer_motion_results.csv"

model = mujoco.MjModel.from_xml_path(str(XML))
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

dt = model.opt.timestep
duration = 5.0
steps = int(duration / dt)

q_addresses = model.jnt_qposadr[:6].astype(int)
v_addresses = model.jnt_dofadr[:6].astype(int)

q_initial = data.qpos[q_addresses].copy()
observers = [
    JointObserver(dt=dt, measurement_std=0.001, acceleration_std=2.0)
    for _ in range(6)
]

rng = np.random.default_rng(21)
records = []

for step in range(steps):
    t = step * dt

    # Smooth shoulder-pan motion: amplitude 0.15 rad, frequency 0.25 Hz.
    q_reference = q_initial.copy()
    q_reference[0] += 0.15 * np.sin(2 * np.pi * 0.25 * t)

    q_true = data.qpos[q_addresses].copy()
    q_measured = q_true + rng.normal(0.0, 0.001, size=6)

    estimates = np.array([
        observers[i].update(q_measured[i]) for i in range(6)
    ])
    qd_estimated = estimates[:, 1]

    # MuJoCo position servos track the commanded joint angles.
    data.ctrl[:6] = np.clip(
        q_reference,
        model.actuator_ctrlrange[:6, 0],
        model.actuator_ctrlrange[:6, 1],
    )

    qd_true_before = data.qvel[v_addresses].copy()
    records.append([
        t, *q_measured, *qd_estimated, *q_true, *qd_true_before
    ])

    mujoco.mj_step(model, data)

records = np.asarray(records)
headers = (
    ["time_s"]
    + [f"q_measured_{i}" for i in range(1, 7)]
    + [f"qd_estimated_{i}" for i in range(1, 7)]
    + [f"q_true_{i}" for i in range(1, 7)]
    + [f"qd_true_{i}" for i in range(1, 7)]
)
np.savetxt(
    OUTPUT, records, delimiter=",",
    header=",".join(headers), comments=""
)

time = records[:, 0]
valid = time >= 0.2
estimated = records[valid, 7:13]
truth = records[valid, 19:25]
error = estimated - truth

rmse = np.sqrt(np.mean(error**2, axis=0))
mae = np.mean(np.abs(error), axis=0)

print("\n========== MOVING-REFERENCE OBSERVER TEST ==========")
print("Duration:", duration, "s")
print("Shoulder-pan reference: amplitude 0.15 rad, frequency 0.25 Hz")
print("Position measurement noise: 0.001 rad")
print("\nJoint | Velocity RMSE (rad/s) | Velocity MAE (rad/s)")
for j in range(6):
    print(f"{j+1:5d} | {rmse[j]:20.6f} | {mae[j]:20.6f}")
print("\nOverall velocity RMSE (rad/s):",
      round(float(np.sqrt(np.mean(error**2))), 6))
print("Saved:", OUTPUT)
print("====================================================\n")
