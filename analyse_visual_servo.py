from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

path = Path("visual_servo_results.csv")
if not path.exists():
    raise SystemExit(f"Missing {path}. Run the demo first.")

df = pd.read_csv(path)
print("\n=== VISUAL SERVO DATASET ===")
print(f"Rows: {len(df)}")
print("Columns:", list(df.columns))
print("\n=== NUMERICAL SUMMARY ===")
print(df.describe().round(4).to_string())

numeric = df.select_dtypes(include="number")
if numeric.empty:
    raise SystemExit("No numeric columns found to plot.")

# Prefer simulation time as the horizontal axis.
time_candidates = [
    c for c in numeric.columns
    if c.lower() in ("time", "t", "time_s", "simulation_time")
]
xcol = time_candidates[0] if time_candidates else numeric.columns[0]

plot_cols = [
    c for c in numeric.columns
    if c != xcol and numeric[c].nunique() > 1
]
if not plot_cols:
    raise SystemExit("No varying numeric columns available to plot.")

fig, axes = plt.subplots(
    len(plot_cols), 1,
    figsize=(10, max(3, 2.5 * len(plot_cols))),
    squeeze=False
)
for ax, col in zip(axes[:, 0], plot_cols):
    ax.plot(numeric[xcol], numeric[col])
    ax.set_ylabel(col)
    ax.grid(True, alpha=0.3)
axes[-1, 0].set_xlabel(xcol)
fig.tight_layout()
out = Path("visual_servo_analysis.png")
fig.savefig(out, dpi=180)
print(f"\nSaved plot: {out.resolve()}")
