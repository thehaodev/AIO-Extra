import numpy as np
import cv2
from matplotlib import pyplot as plt


def gaussian_kernel(size, sigma):
    if size % 2 == 0:
        size = size + 1

    max_point = size // 2
    min_point = -max_point
    K = np.zeros((size, size))
    for x in range(min_point, max_point + 1):
        for y in range(min_point, max_point + 1):
            sigma_p = np.power(sigma, 2)
            x_p = np.power(x, 2)
            y_p = np.power(y, 2)
            value = 1/(2*np.pi*sigma_p) * np.exp(-(x_p+y_p)/(2*sigma_p))
            K[x - min_point, y - min_point] = value

    return K


def run():
    kernel = gaussian_kernel(5, 10.0)
    img = cv2.imread("2.jpg", 0)
    img_gaussian = cv2.filter2D(img, -1, kernel)
    plt.imshow(img_gaussian, cmap="gray")
    plt.show()


run()
