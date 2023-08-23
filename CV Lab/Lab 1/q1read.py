import cv2 as cv
img = cv.imread("../unsplash.jpg", 0)
cv.imshow('unsp', img)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imwrite('unsp.jpg',img)
