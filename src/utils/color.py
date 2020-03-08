import cv2

def convert_from_bgr(image, color_space):
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

    return cv2.cvtColor(image, color)