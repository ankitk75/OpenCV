import cv2 as cv
import numpy as np
import os
import glob

CHECKBOARD = (12, 12)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 50, 0.0001)

objpoints = []
imgpoints = []

objp = np.zeros((1, CHECKBOARD[0] * CHECKBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKBOARD[0], 0:CHECKBOARD[1]].T.reshape(-1,2)

prev_img_shape = None

images = glob.glob('calibrate/*.tif')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, CHECKBOARD, cv.CALIB_CB_ADAPTIVE_THRESH +
                                           cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)

    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        img = cv.drawChessboardCorners(img, CHECKBOARD, corners2, ret)

    cv.imshow("img", img)
    cv.waitKey(0)

cv.destroyAllWindows()

h, w = img.shape[:2]

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],
                                                  None, None)

print("Camera matrix : \n")
print(mtx)

print("dist: \n")
print(dist)

print("rvecs: \n")
print(rvecs)

print("tvecs: \n")
print(tvecs)


# Intrinsic parameters
focal_length = (mtx[0, 0] + mtx[1, 1]) / 2
principal_point = (mtx[0, 2], mtx[1, 2])

print("Focal Length: ", focal_length)
print("Principal Point: ", principal_point)

# Extrinsic parameters (using solvePnP)
for i in range(len(objpoints)):
    _, rvec, tvec, inliers = cv.solvePnPRansac(objpoints[i], imgpoints[i], mtx, dist)

    print(f"Extrinsic Parameters for image {i + 1}:")
    print("Rotation Vector:")
    print(rvec)
    print("Translation Vector:")
    print(tvec)
    print("---------------------------")

    # Reproject points
    imgpoints_reprojected, _ = cv.projectPoints(objpoints[i], rvec, tvec, mtx, dist)
    imgpoints_reprojected = np.squeeze(imgpoints_reprojected)

    # Draw reprojected points on the image
    for p1, p2 in zip(imgpoints[i], imgpoints_reprojected):
        p1 = tuple(map(int, p1.ravel()))
        p2 = tuple(map(int, p2.ravel()))
        cv.line(img, p1, p2, (0, 255, 0), 2)

    cv.imshow("Reprojected Points", img)
    cv.waitKey(0)

cv.destroyAllWindows()


