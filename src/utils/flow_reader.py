import cv2
import numpy as np


def read_flow_field(path: str) -> np.ndarray:
    """
    Method based on the provided code from KITTI (flow_read.m)

    Optical flow maps are saved as 3-channel uint16 PNG images: The first channel
    contains the u-component, the second channel the v-component and the third
    channel denotes if a valid ground truth optical flow value exists for that
    pixel (1 if true, 0 otherwise)
    """

    im = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.double)

    # (u,v) flow vector
    im_u = (im[:, :, 2] - 2 ** 15) / 64
    im_v = (im[:, :, 1] - 2 ** 15) / 64

    # pixel exists or not
    im_valid = im[:, :, 0]
    im_valid[im_valid > 1] = 1

    im_u[im_valid == 0] = 0
    im_v[im_valid == 0] = 0

    flow_field = np.dstack((im_u, im_v, im_valid))

    return flow_field


def read_grayscale_image(path: str) -> np.ndarray:
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
