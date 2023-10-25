import cv2 as cv
import numpy as np

def HarrisCorner(img, k, thresh):
    img_cpy = np.copy(img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    Ix = cv.Sobel(gray, cv.CV_64F, 1, 0)
    Iy = cv.Sobel(gray, cv.CV_64F, 0, 1)

    Ix2 = np.square(Ix)
    Iy2 = np.square(Iy)
    Ixy = Ix*Iy

    cv.normalize(harris, harris, 0, 1, cv.NORM_MINMAX)

    loc = np.where(harris >= thresh)
    for pt in zip(*loc[::-1]):
        cv.circle(img_cpy, pt, 2, (0, 0, 255), 1)

    return img_cpy

img = cv.imread('chess.jpg')
cv.imshow('OG', img)

harris_corner = HarrisCorner(img, 0.045, 0.35)
cv.imshow('Harris Corner', harris_corner)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
dst = cv.cornerHarris(gray,2,5,0.06)
dst = cv.dilate(dst,None)
img[dst>0.04*dst.max()] = [255,0,0]
cv.imshow('Using func', img)

cv.waitKey(0)