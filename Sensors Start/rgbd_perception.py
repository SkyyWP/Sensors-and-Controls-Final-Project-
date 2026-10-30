import numpy as np


# ---------------------------------------------------------
# CAMERA INTRINSICS
# ---------------------------------------------------------

def calculate_intrinsics(width, height, vertical_fov_deg):
    """
    Calculate pinhole camera intrinsics from image dimensions
    and vertical field of view.

    Returns:
        Dictionary containing fx, fy, cx, cy.
    """

    fovy = np.deg2rad(vertical_fov_deg)

    fy = height / (2.0 * np.tan(fovy / 2.0))
    fx = fy

    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    return {
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "width": int(width),
        "height": int(height)
    }


# ---------------------------------------------------------
# TARGET DETECTION
# ---------------------------------------------------------

def detect_red_target(rgb, min_pixels=20):
    """
    Detect a red target in an RGB image.

    Args:
        rgb:
            RGB image with shape (height, width, 3).

        min_pixels:
            Minimum number of red pixels required.

    Returns:
        Dictionary containing:
            u              Median horizontal pixel coordinate
            v              Median vertical pixel coordinate
            pixel_count    Number of detected red pixels
            mask           Binary detection mask

        Returns None if no valid target is detected.
    """

    rgb = np.asarray(rgb)

    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("RGB image must have shape (height, width, 3).")

    red = (
        (rgb[:, :, 0] > 120)
        & (rgb[:, :, 0] > 1.8 * rgb[:, :, 1])
        & (rgb[:, :, 0] > 1.8 * rgb[:, :, 2])
    )

    ys, xs = np.where(red)

    if len(xs) < min_pixels:
        return None

    return {
        "u": float(np.median(xs)),
        "v": float(np.median(ys)),
        "pixel_count": int(len(xs)),
        "mask": red
    }


# ---------------------------------------------------------
# DEPTH LOCALISATION
# ---------------------------------------------------------

def estimate_target_position(
    rgb,
    depth,
    camera_to_world,
    intrinsics,
    depth_scale=1.0,
    min_pixels=20,
    min_depth=0.02,
    max_depth=2.0
):
    """
    Detect a red target and estimate its 3D world position.

    Args:
        rgb:
            RGB image (height, width, 3).

        depth:
            Depth image (height, width).

        camera_to_world:
            4x4 transformation matrix describing the camera
            optical frame relative to the world frame.

            An SE3 object is also accepted.

        intrinsics:
            Dictionary containing fx, fy, cx and cy.

        depth_scale:
            Multiplier converting raw depth values into metres.

            Examples:
                Depth already in metres: 1.0
                Depth in millimetres:    0.001
                RealSense raw Z16:       usually 0.001,
                                        subject to device settings.

        Returns:
            Dictionary containing:
                target_world
                target_camera
                pixel
                depth_m
                pixel_count
                confidence

            Returns None if detection or depth estimation fails.
    """

    rgb = np.asarray(rgb)
    depth = np.asarray(depth)

    # Validate image dimensions.
    if rgb.shape[:2] != depth.shape[:2]:
        raise ValueError(
            "RGB and depth images must have matching dimensions."
        )

    if depth.ndim != 2:
        raise ValueError("Depth image must be a 2D array.")

    # Check camera intrinsics.
    required = ("fx", "fy", "cx", "cy")

    if not all(key in intrinsics for key in required):
        raise ValueError(
            "Intrinsics must contain fx, fy, cx and cy."
        )

    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])

    if fx <= 0 or fy <= 0:
        raise ValueError("Camera focal lengths must be positive.")

    # Detect the target in the RGB image.
    detection = detect_red_target(rgb, min_pixels)

    if detection is None:
        return None

    mask = detection["mask"]

    # Extract depth values at detected red pixels.
    valid_depth = (
        np.isfinite(depth)
        & (depth > 0)
    )

    valid_mask = mask & valid_depth

    ys, xs = np.where(valid_mask)

    if len(xs) < min_pixels:
        return None

    # Convert depth into metres.
    depth_values = depth[ys, xs].astype(float) * depth_scale

    # Reject invalid or out-of-range depth measurements.
    good = (
        np.isfinite(depth_values)
        & (depth_values > min_depth)
        & (depth_values < max_depth)
    )

    if np.count_nonzero(good) < min_pixels:
        return None

    xs = xs[good]
    ys = ys[good]
    depth_values = depth_values[good]

    # Median pixel and median depth.
    u = float(np.median(xs))
    v = float(np.median(ys))
    depth_m = float(np.median(depth_values))

    # -----------------------------------------------------
    # PIXEL TO CAMERA COORDINATES
    # -----------------------------------------------------

    # Optical camera frame:
    # X = right, Y = down, Z = forward.

    target_camera = np.array([
        (u - cx) * depth_m / fx,
        (v - cy) * depth_m / fy,
        depth_m
    ])

    # -----------------------------------------------------
    # CAMERA TO WORLD COORDINATES
    # -----------------------------------------------------

    # Accept either a NumPy 4x4 matrix or a Spatialmath SE3.
    if hasattr(camera_to_world, "A"):
        T_world_camera = np.asarray(camera_to_world.A, dtype=float)
    else:
        T_world_camera = np.asarray(camera_to_world, dtype=float)

    if T_world_camera.shape != (4, 4):
        raise ValueError("camera_to_world must be a 4x4 matrix.")

    target_camera_h = np.append(target_camera, 1.0)

    target_world_h = T_world_camera @ target_camera_h
    target_world = target_world_h[:3]

    return {
        "target_world": target_world,
        "target_camera": target_camera,
        "pixel": (u, v),
        "depth_m": depth_m,
        "pixel_count": int(len(xs)),
        "confidence": float(len(xs) / max(detection["pixel_count"], 1))
    }
