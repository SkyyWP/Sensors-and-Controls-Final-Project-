import numpy as np

A = np.eye(3) 
C = np.eye(3) 


def camera_Rn(cam_rot, depth, fx, sigma_px=1.0, sigma_d0=0.002, sigma_dk=0.005):
    #cam_rot 3x3 rotation from camera frame to world frame (camera's orientation)
    #depth     measured depth [m]
    #fx        focal length [pixels]
    s_side = sigma_px * depth / fx
    s_depth = sigma_d0 + sigma_dk * depth**2
    Rn_cam = np.diag([s_side**2, s_side**2, s_depth**2])
    return cam_rot @ Rn_cam @ cam_rot.T                       # rotate into the world frame


def kf_predict(xh, P, Qd):
    xh_pred = A @ xh
    P_pred = A @ P @ A.T + Qd      
    return xh_pred, P_pred


def kf_update(xh_pred, P_pred, y, Rn):
    S = C @ P_pred @ C.T + Rn            # innovation covariance
    K = P_pred @ C.T @ np.linalg.inv(S)  # Kalman gain
    nu = y - C @ xh_pred                 # innovation
    xh = xh_pred + K @ nu
    P = (np.eye(3) - K @ C) @ P_pred     
    P = 0.5 * (P + P.T)
    return xh, P, nu, S


def kf_step(xh, P, y, Qd, Rn):
    xh, P = kf_predict(xh, P, Qd)
    if y is None:                        # no measurement: keep the prediction
        return xh, P, None
    xh, P, nu, S = kf_update(xh, P, y, Rn)
    nis = float(nu @ np.linalg.solve(S, nu))
    return xh, P, nis
