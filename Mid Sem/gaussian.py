import cv2
import numpy as np

def gaussian_blur(image, kernel_size, sigma):
    # Create a Gaussian kernel
    kernel = np.fromfunction(
        lambda x, y: (1/ (2 * np.pi * sigma ** 2)) * np.exp(-((x - (kernel_size // 2)) ** 2 + (y - (kernel_size // 2)) ** 2) / (2 * sigma ** 2)),
        (kernel_size, kernel_size)
    )
    kernel /= np.sum(kernel)  # Normalize the kernel

    # Get image dimensions
    height, width = image.shape

    # Initialize the output image for the blurred result
    blurred_image = np.zeros((height, width), dtype=np.uint8)

    # Perform convolution
    for y in range(kernel_size // 2, height - kernel_size // 2):
        for x in range(kernel_size // 2, width - kernel_size // 2):
            neighborhood = image[y - kernel_size // 2:y + kernel_size // 2 + 1, x - kernel_size // 2:x + kernel_size // 2 + 1]
            blurred_image[y, x] = np.sum(neighborhood * kernel)

    return blurred_image

def box_filter(image, kernel_size):
    # Create a kernel with equal weights
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)

    # Get image dimensions
    height, width = image.shape

    # Initialize the output image for the filtered result
    filtered_image = np.zeros((height, width), dtype=np.uint8)

    # Perform convolution
    for y in range(kernel_size // 2, height - kernel_size // 2):
        for x in range(kernel_size // 2, width - kernel_size // 2):
            neighborhood = image[y - kernel_size // 2:y + kernel_size // 2 + 1, x - kernel_size // 2:x + kernel_size // 2 + 1]
            filtered_image[y, x] = np.sum(neighborhood * kernel)

    return filtered_image

if __name__ == "__main__":
    input_image_path = "input_image.jpg"  # Replace with your input image path
    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

    kernel_size = 5  # Adjust the kernel size as needed
    sigma = 1.0      # Adjust the standard deviation (sigma) as needed

    blurred_image = gaussian_blur(input_image, kernel_size, sigma)

    # Display the blurred image (you can save or process it as needed)
    cv2.imshow("Gaussian Blurred Image", blurred_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
