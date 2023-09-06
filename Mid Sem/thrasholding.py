import cv2
import numpy as np

def simple_thresholding(image, threshold):
    # Get image dimensions
    height, width = image.shape

    # Initialize the output binary image
    binary_image = np.zeros((height, width), dtype=np.uint8)

    # Iterate through the image pixels and apply thresholding
    for y in range(height):
        for x in range(width):
            if image[y, x] >= threshold:
                binary_image[y, x] = 255  # Set pixel to white (255)
            else:
                binary_image[y, x] = 0    # Set pixel to black (0)

    return binary_image

if __name__ == "__main__":
    input_image_path = "input_image.jpg"  # Replace with your input image path
    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

    threshold_value = 128  # Adjust the threshold value as needed

    binary_image = simple_thresholding(input_image, threshold_value)

    # Display the binary image (you can save or process it as needed)
    cv2.imshow("Binary Image (Simple Thresholding)", binary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
