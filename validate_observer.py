from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
CSV = HERE / "joint_control_results.csv"
OUTPUT = HERE / "observer_validation_summary.csv"

data = np.genfromtxt(CSV, delimiter=",", names=True)

time = data["time_s"]
joint_count = 6

print("\n========== OBSERVER VALIDATION ==========")

# Match each estimated velocity with the true velocity recorded
# at the same simulation step.
valid = time >= 0.2

rmse = []
mae = []

for j in range(1, joint_count + 1):
    estimated = data[f"qd_estimated_{j}"][valid]
    truth = data[f"qd_true_{j}"][valid]
    error = estimated - truth

    rmse.append(np.sqrt(np.mean(error**2)))
    mae.append(np.mean(np.abs(error)))

print("Evaluation window: 0.2 s to", round(float(time[-1]), 3), "s")
print("\nJoint | Velocity RMSE (rad/s) | Velocity MAE (rad/s)")
for j, (r, m) in enumerate(zip(rmse, mae), start=1):
    print(f"{j:5d} | {r:20.6f} | {m:20.6f}")

overall_rmse = np.sqrt(np.mean(np.asarray(rmse)**2))
print("\nOverall velocity RMSE (rad/s):", round(float(overall_rmse), 6))

summary = np.column_stack((np.arange(1, 7), rmse, mae))
np.savetxt(
    OUTPUT,
    summary,
    delimiter=",",
    header="joint,velocity_rmse_rad_s,velocity_mae_rad_s",
    comments="",
    fmt=["%d", "%.8f", "%.8f"],
)

print("Summary saved to:", OUTPUT)
print("========================================\n")
