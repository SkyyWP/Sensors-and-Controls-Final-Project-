from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "ur3e_scene" / "scene_pick.xml"

CAMERA_NAME = "wrist_rgbd"
TARGET_BODY = "target"
TARGET_JOINT = "target_free"
END_EFFECTOR_SITE = "attachment_site"

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# Camera / perception
MIN_RED_PIXELS = 25
RED_THRESHOLD = 120
RED_RATIO = 1.8
SURFACE_TO_CENTRE = 0.030

# Target estimator
TARGET_PROCESS_NOISE = 1e-3 #0.01 
TARGET_MEASUREMENT_NOISE = 0.01  # 9e-6
CAMERA_NOISE = True # Camera noise (camera_noise.py), False for no noise

# Control
IK_DAMPING = 0.04
IK_MAX_ITERATIONS = 120
IK_STEP_LIMIT = 0.5  # 0.12

APPROACH_HEIGHT = 0.12
GRASP_HEIGHT = 0.045
LIFT_HEIGHT = 0.12

# Simulation
CAPTURE_INTERVAL = 100  # 50
SCAN_STEPS = 5000  # 350
MAX_APPROACH_STEPS = 15000
RANDOM_SEED = 42
