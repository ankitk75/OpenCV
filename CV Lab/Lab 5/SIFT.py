import cv2 as cv

# Important NOTE: Use opencv <= 3.4.2.16 as
# SIFT is no longer available in
# opencv > 3.4.2.16
# import cv2

# Loading the image
img = cv.imread('../pic/1.jpg')

# Converting image to grayscale
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# Applying SIFT detector
sift = cv.SIFT_create()
kp = sift.detect(gray, None)

# Marking the keypoint on the image using circles
img=cv.drawKeypoints(gray, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# cv.imwrite('image-with-keypoints.jpg', img)
cv.imshow('image=with-keypoints', img)
cv.waitKey(0)