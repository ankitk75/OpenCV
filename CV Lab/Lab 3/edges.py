import cv2

original_image = cv2.imread('../pic/7.jpg')

if original_image is None:
    print("Image not found.")
else:
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)

    # Display the original image and the detected edges
    cv2.imshow('Original Image', original_image)
    cv2.imshow('Detected Edges (Canny)', edges)

    # Wait for a key press and then close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# img0 = cv2.imread('../pic/7.jpg',)
#
# # converting to gray scale
# gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
#
# # remove noise
# img = cv2.GaussianBlur(gray,(3,3),0)
#
# # convolute with proper kernels
# laplacian = cv2.Laplacian(img,cv2.CV_64F)
#
#
# plt.imshow(laplacian,cmap = 'gray')
# plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
#
#
# plt.show()