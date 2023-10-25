import numpy as np
import cv2 as cv
import glob

square_size = 0.025
CHESSBOARD = (12, 12)

object_points = []
image_points = []

objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2) * square_size

images = glob.glob('./camcal*.tif')
img = None

for image_file in images:

    img = cv.imread(image_file, 0)
    ret, corners = cv.findChessboardCorners(img, (CHESSBOARD[0], CHESSBOARD[1]), None)

    if ret:
        object_points.append(objp)
        image_points.append(corners)

ret, camera_matrix, distortion_coefficients, rvecs, tvecs = cv.calibrateCamera(
    object_points, image_points, img.shape[::-1], None, None
)

print(f"Camera Matrix:" )
print(camera_matrix)

print(f"\nDistortion Coefficients:")
print(distortion_coefficients)

print(f"\nRotation Vectors: ", end='\n')
for i, v in enumerate(rvecs):
    print(f"\nImg {i+1}:")
    print(v)

print(f"\nTranslation Vectors:", end='\n')
for i, v in enumerate(tvecs):
    print(f"\nImg {i+1}:")
    print(v)


distorted_img = cv.imread("Resources/distortion.tif", 0)
new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(
    camera_matrix, distortion_coefficients, distorted_img.shape[::-1], 1, distorted_img.shape[::-1]
)

undistorted = cv.undistort(distorted_img, camera_matrix, distortion_coefficients, None, new_camera_matrix)
x, y, w, h = roi
undistorted = undistorted[y:y + h, x:x + w]

cv.imshow("Distorted", distorted_img)
cv.imshow("Undistorted", undistorted)
cv.waitKey(0)