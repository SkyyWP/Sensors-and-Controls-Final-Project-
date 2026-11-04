import numpy as np
import cv2 as cv
import mujoco
from config import MODEL_PATH
from rgbd_perception import RGBDPerception

#MAIN CODE GATHERED FROM OPEN CV AND ADAPTED https://docs.opencv.org/3.4.20/dc/dbb/tutorial_py_calibration.html
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((8*6,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2) * 15 #times the square size 

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

#Import the scene so it reads off of the scene instead of images
model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)
camera = RGBDPerception(model)
q_home = data.qpos[:6].copy()


while len(objpoints) < 15: #checks 15 photos from the wrist camera 
    img = cv.cvtColor(camera.capture(data)["rgb"], cv.COLOR_RGB2BGR)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (8,6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (5,5), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (8,6), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)
cv.destroyAllWindows()
camera.close()

flags = cv.CALIB_ZERO_TANGENT_DIST | cv.CALIB_FIX_K1 | cv.CALIB_FIX_K2 | cv.CALIB_FIX_K3
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None, flags=flags)


mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i].reshape(-1, 2), imgpoints2.reshape(-1, 2), cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )

#Save the result to let rgbd load K
np.savez('calibration.npz', K=mtx, dist=dist)