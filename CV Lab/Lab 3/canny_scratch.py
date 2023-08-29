import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("Resources/1.jpg", 0)
# img = cv.GaussianBlur(img, (5, 5), 0)

sobel_x = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
sobel_y = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)

mag, ang = cv.cartToPolar(sobel_x, sobel_y, angleInDegrees=True)

mag_max = np.max(mag)
mag = mag / mag_max * 255

low = 50
high = 254

height, width = img.shape
new_img = np.zeros((height, width))

strong = np.where(mag >= high)
non_edge = np.where(mag < low)
weak = np.where((mag < high) & (mag >= low))

new_img[strong] = 255
new_img[non_edge] = 0
new_img[weak] = 128

to_convert = []
for i in range(1, height - 1):
    for j in range(1, width - 1):
        if new_img[i, j] == 128:
            if new_img[i + 1, j] == 255 or new_img[i - 1, j] == 255 or new_img[i, j + 1] == 255 or new_img[
                i, j - 1] == 255 or new_img[i + 1, j + 1] == 255 or new_img[i - 1, j - 1] == 255 or new_img[
                i + 1, j - 1] == 255 or new_img[i - 1, j + 1] == 255:
                to_convert.append([i, j])

for i, j in to_convert:
    new_img[i, j] = 255

cv.imshow("Canny", new_img)
cv.imshow("Original", img)
cv.imshow("Inbuilt", cv.Canny(img, 100, 200))
cv.waitKey(0)

cv.destroyAllWindows()
