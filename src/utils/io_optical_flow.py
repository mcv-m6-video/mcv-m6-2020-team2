import cv2
import numpy as np
from matplotlib import pyplot as plt

def read_flow_field(path:str, frame_id:str, plot:bool=False) -> np.ndarray:
    """
    Method based on the provided code from KITTI (flow_read.m)

    Optical flow maps are saved as 3-channel uint16 PNG images: The first channel
    contains the u-component, the second channel the v-component and the third
    channel denotes if a valid ground truth optical flow value exists for that
    pixel (1 if true, 0 otherwise)
    """
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.double)

    # (u,v) flow vector
    im_u = (im[:, :, 2] - 2**15) / 64
    im_v = (im[:, :, 1] - 2**15) / 64

    # pixel exists or not
    im_valid = im[:, :, 0]
    im_valid[im_valid>1] = 1

    im_u[im_valid == 0] = 0
    im_v[im_valid == 0] = 0

    flow_field = np.dstack((im_u, im_v, im_valid))

    if plot:
        plt.imshow((flow_field * 255).astype(np.uint8))
        plt.axis('off')
        plt.title(f'Flow Field {frame_id}')
        plt.show()

    return flow_field

def read_grayscale_image(path:str, frame_id:str, plot:bool=False) -> np.ndarray:
    im = cv2.imread(path, 0)
    if plot:
        plt.imshow(im, cmap='gray')
        plt.axis('off')
        plt.title(f'Image Sequence {frame_id}')
        plt.show()

    return  im



