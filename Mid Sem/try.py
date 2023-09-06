import cv2
import numpy as np

def blur(image):

    gauss = np.array([[1, 2, 1],
                                [2, 4, 2],
                                [1, 2, 1]], dtype=float)

    height, width = image.shape

    res = np.zeros((height, width), dtype=float)

    for y in range(1, height - 1):
        for x in range(1, width - 1):

            res = np.sum(gauss * image[y - 1:y + 2, x - 1:x + 2])

    return res

if __name__ == "__main__":
    input_image_path = "../../Resources/7.jpg"  # Replace with your input image path
    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

    gb = blur(input_image)

    # Display the gradient magnitude and direction (you can save or process them as needed)
    cv2.imshow("gauss", gb)
    # cv2.imshow("Gradient Direction", gradient_direction.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
