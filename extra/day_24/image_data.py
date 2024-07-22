import numpy as np
import cv2
from matplotlib import pyplot as plt


def read_image(path):
    return cv2.imread(path, 1)


def adjust_brightness(image, value):
    new_img = image.astype(np.float32) + value
    new_img = np.clip(new_img, 0, 255)
    new_img = new_img.astype(np.uint8)

    return new_img


def change_color_channel(image, channel):
    return cv2.cvtColor(image, channel)


def crop_image(image, margin: int):
    shape = image.shape
    half_width = int(shape[0] / 2)
    half_height = int(shape[1] / 2)

    return image[half_width:half_width + margin, half_height:half_height + margin]


def run():
    image = read_image("1.jpg")

    # gray image
    _ = change_color_channel(image, cv2.COLOR_RGB2GRAY)

    # brightness image
    _ = adjust_brightness(image, 50)

    # darkness image
    _ = adjust_brightness(image, -80)

    # crop image
    _ = crop_image(image, 1000)

    plt.imshow(_, cmap="gray")
    plt.show()


run()
