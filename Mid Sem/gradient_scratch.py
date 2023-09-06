import cv2
import numpy as np

def gradient(image):

    kernel_x = np.array([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=float)

    kernel_y = np.array([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]], dtype=float)

    height, width = image.shape

    gradient_mag = np.zeros((height, width), dtype=float)
    gradient_dir = np.zeros((height, width), dtype=float)

    for y in range(1, height - 1):
        for x in range(1, width - 1):

            gx = np.sum(kernel_x * image[y - 1:y + 2, x - 1:x + 2])
            gy = np.sum(kernel_y * image[y - 1:y + 2, x - 1:x + 2])

            gradient_mag[y, x] = np.sqrt(gx**2 + gy**2)

            gradient_dir[y, x] = np.arctan2(gy, gx) * (180 / np.pi)

    return gradient_mag, gradient_dir

if __name__ == "__main__":
    input_image_path = "../../Resources/7.jpg"
    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

    gradient_magnitude, gradient_direction = gradient(input_image)

    cv2.imshow("Gradient Magnitude", gradient_magnitude.astype(np.uint8))
    cv2.imshow("Gradient Direction", gradient_direction.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
