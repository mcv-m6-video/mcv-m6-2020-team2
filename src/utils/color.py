import cv2


def convert_from_bgr(image, color_space, channels=None):
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

    converted = cv2.cvtColor(image, color)
    if channels is not None:
        converted = converted[:, :, channels]

    num_channels = default_num_channels(color_space) if channels is None else len(channels)
    return converted.reshape(image.shape[0], image.shape[1], num_channels)


def default_num_channels(color_space):
    color_space = color_space.lower()

    if color_space == 'gray':
        return 1
    elif color_space in ['hsv', 'lab', 'yuv', 'rgb', 'bgr']:
        return 3
    else:
        raise ValueError(f'Unknown color space: {color_space}')
