import cv2 as cv
img = cv.imread("../unsplash.jpg", 0)
color = img[10,20]
blue = int(color[0])
green = int(color[1])
red = int(color[2])
print(f'Blue: {blue}\n Green: {green}\n Red: {red}')