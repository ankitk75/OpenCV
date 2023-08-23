import cv2 as cv

img = cv.imread('../pic/7.jpg')

t_lower = 50
t_upper = 150

edge = cv.Canny(img, t_lower, t_upper)

cv.imshow('original', img)
cv.imshow('edge', edge)
cv.waitKey(0)
cv.destroyAllWindows()