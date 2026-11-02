#Intergrating IBVS

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

from spatialmath import SE3, SO3
import roboticstoolbox as rtb
from machinevisiontoolbox import Image, CentralCamera, IBVS, PBVS, mkgrid

def tag_corners(s):
    '''Corners of a square tag of side s in the tag frame, in ArUco order
    (top-left, top-right, bottom-right, bottom-left). The tag z-axis points out of the face.'''
    return np.array([[-s/2,  s/2, 0],
                     [ s/2,  s/2, 0],
                     [ s/2, -s/2, 0],
                     [-s/2, -s/2, 0]]).T
    
ur5  = rtb.models.DH.UR5()
T_EC = SE3.Tz(0.05)                       # camera 5 cm out along the tool z-axis, optical axis = tool z
ur5.tool = T_EC

wrist = CentralCamera(f=600, imagesize=(640, 480), rho=1, name="wrist camera")
S_TAB = 0.10
T_W_tag = SE3(-0.45, -0.15, 0.0) * SE3.Rz(0.6)          # tag lying on the table, z up
P_tab   = T_W_tag * tag_corners(S_TAB)                  # 3 x 4 world points
pd_arm  = np.c_[wrist.pp] + 120 * np.array([[-1, 1, 1, -1], [-1, -1, 1, 1]])   # 240 px square, centred

q0 = np.array([0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0])   # camera looking straight down
print(ur5)
print("camera pose at q0:\n", ur5.fkine(q0))
uv0 = wrist.project_point(P_tab, pose=ur5.fkine(q0))
print("initial tag corners in the wrist image:\n", uv0)
print("desired depth for a 240 px tag: %.3f m" % (wrist.f[0] * S_TAB / 240 if np.ndim(wrist.f) else wrist.f * S_TAB / 240))

# quick check that jacobe really is the camera-frame Jacobian
dq = 1e-6 * np.array([1, -2, 3, 1, 2, -1])
twist_fd = (ur5.fkine(q0).inv() * ur5.fkine(q0 + dq)).delta()
print("jacobe check (should be ~0):", np.abs(ur5.jacobe(q0) @ dq - twist_fd).max())


    
def run_arm_ibvs(lam=1.0, dt=0.05, nsteps=400, depth=None):
    q = q0.copy()
    log = {"q": [], "uv": [], "e": [], "qd": [], "manip": []}
    for k in range(nsteps):
        T_cam = ur5.fkine(q)                                   # camera pose (tool = camera)
        uv = wrist.project_point(P_tab, pose=T_cam)            # current features
        Z = (T_cam.inv() * P_tab)[2] if depth is None else depth

        J_img = wrist.visjac_p(uv, Z)
        nu = lam * np.linalg.pinv(J_img) @ (pd_arm - uv).flatten(order="F")
        qd = np.linalg.pinv(ur5.jacobe(q)) @ nu

        e = (pd_arm - uv).flatten(order="F")
        log["q"].append(q.copy()); log["uv"].append(uv); log["e"].append(np.linalg.norm(e))
        log["qd"].append(qd); log["manip"].append(ur5.manipulability(q))
        if np.linalg.norm(e) < 0.5:
            break
        q = q + qd * dt
    return {k: np.array(v) for k, v in log.items()}

arm = run_arm_ibvs()
print(f"{len(arm['e'])} steps, final error {arm['e'][-1]:.2f} px")
print("final camera position (m):", ur5.fkine(arm["q"][-1]).t)