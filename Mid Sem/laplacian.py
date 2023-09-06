import cv2
import numpy as np

def laplacian_gradient(image):
    # Define the Laplacian kernel
    laplacian_kernel = np.array([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]])

    # Get image dimensions
    height, width = image.shape

    # Initialize the output image for the Laplacian gradient
    gradient_image = np.zeros((height, width), dtype=np.float32)

    # Perform convolution
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            neighborhood = image[y - 1:y + 2, x - 1:x + 2]
            gradient_image[y, x] = np.sum(neighborhood * laplacian_kernel)

    return gradient_image

if __name__ == "__main__":
    input_image_path = "../../Resources/4.jpg"  # Replace with your input image path
    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

    laplacian_grad = laplacian_gradient(input_image)

    # Display the Laplacian gradient (you can save or process it as needed)
    cv2.imshow("Laplacian Gradient", laplacian_grad.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # sobelx = cv.Sobel(img, cv.CV_64F, 1, 0)
    # sobely = cv.Sobel(img, cv.CV_64F, 0, 1)
    # sobelx = np.uint8(np.absolute(sobelx))
    # sobely = np.uint8(np.absolute(sobely))
    # sobelcombined = cv.bitwise_or(sobelx, sobely)
    # cv.imshow('OG', img)
    # cv.imshow('Sobel X', sobelx)
    # cv.imshow('Sobel Y', sobely)
    # cv.imshow('Sobel Combined', sobelcombined)
    cv.waitKey(0)

