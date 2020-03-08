import cv2
import numpy as np

def convert_from_bgr(image, color_space, img_reshape):
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
    img = img_reshape(img)

    if len(img.shape) == 2:
        img = img[..., np.newaxis]

    return img


def num_channels(color_space, reshape_channels):
    return convert_from_bgr(np.zeros((1, 1, 3), np.uint8), color_space, reshape_channels).shape[2]