import cv2 as cv
import numpy as np
import matplotlib.pyplot as plot
image = cv.imread('../pic/7.jpg',0)

lap = cv.Laplacian(image, cv.CV_64F, ksize=3)
lap = np.uint8(np.absolute(lap))

sobelx = cv.Sobel(image, 0, dx=1, dy=0)
sobelx = np.uint8(np.absolute(sobelx))

sobely = cv.Sobel(image, 0, dx=0, dy=1)
sobely = np.uint8(np.absolute(sobely))

results = [lap, sobelx, sobely]
images = ["Gradient Img", "Gradient_X", "Gradient_Y"]
plot.figure(figsize=(10, 10))
for i in range(3):
    plot.title(results[i])
    plot.subplot(1, 3, i + 1)
    plot.imshow(results[i], "plasma")
    plot.xticks([])
    plot.yticks([])

plot.show()