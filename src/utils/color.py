import cv2
import numpy as np

def convert_from_bgr(image, color_space, channel_selector):
    color_space = color_space.lower()

    if color_space == 'gray':
       color = cv2.COLOR_BGR2GRAY
    elif color_space == 'hsv':
       color = cv2.COLOR_BGR2HSV
    elif color_space == 'lab':
        color = cv2.COLOR_BGR2LAB
    elif color_space == 'yuv':
        color = cv2.COLOR_BGR2YUV
    elif color_space == 'rgb':
        color = cv2.COLOR_BGR2RGB
    elif color_space == 'bgr':
        return image
    else:
        raise ValueError(f'Unknown color space: {color_space}')

    img = cv2.cvtColor(image, color)
    return channel_selector(img) if len(img.shape) > 2 else img[..., np.newaxis]


def num_channels(color_space, channel_selector):
    if color_space == 'gray':
        return 1
    else:
        return channel_selector(np.zeros((1, 1, 3), np.uint8)).shape[2]