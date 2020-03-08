import cv2
import numpy as np

def convert_color(image, color_space, img_reshape=lambda img: img):
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
        raise ValueError(f"Bad color space: {color_space}")

    img = cv2.cvtColor(image, color)
    img = img_reshape(img)

    if len(img.shape) == 2:
        img = img[..., np.newaxis]

    return img