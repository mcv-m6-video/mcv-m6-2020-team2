import numpy as np
import cv2
from tqdm import trange


def distance(x1: np.ndarray, x2: np.ndarray):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def block_matching_flow(img_prev: np.ndarray, img_next: np.ndarray, block_size=16, search_area=16, motion_type='backward'):
    if motion_type == 'forward':
        reference = img_prev
        target = img_next
    elif motion_type == 'backward':
        reference = img_next
        target = img_prev
    else:
        raise ValueError(f'Unknown motion type {motion_type}')

    assert (reference.shape == target.shape)
    height, width = reference.shape[:2]

    flow_field = np.zeros((height, width, 2), dtype=float)
    for i in trange(0, height - block_size, block_size):
        for j in range(0, width - block_size, block_size):
            min_dist = np.inf
            displacement = [0, 0]
            for ii in range(max(i - search_area, 0), min(i + search_area, height - block_size)):
                for jj in range(max(j - search_area, 0), min(j + search_area, width - block_size)):
                    dist = distance(reference[i:i + block_size, j:j + block_size],
                                    target[ii:ii + block_size, jj:jj + block_size])
                    if dist < min_dist:
                        displacement = [jj - j, ii - i]
                        min_dist = dist
            flow_field[i:i + block_size, j:j + block_size, :] = displacement

    return flow_field


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(v * 4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def read_flow(filename):
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    fx = (img[:, :, 2].astype(float) - 2 ** 15) / 64
    fy = (img[:, :, 1].astype(float) - 2 ** 15) / 64
    f_valid = img[:, :, 0].astype(bool)
    fx[~f_valid] = 0
    fy[~f_valid] = 0
    flow = np.dstack((fx, fy, f_valid))
    return flow


if __name__ == '__main__':
    img_prev = cv2.imread('../../data/data_stereo_flow/training/image_0/000045_10.png', cv2.IMREAD_GRAYSCALE)
    img_next = cv2.imread('../../data/data_stereo_flow/training/image_0/000045_11.png', cv2.IMREAD_GRAYSCALE)
    flow_noc = read_flow('../../data/data_stereo_flow/training/flow_noc/000045_10.png')

    flow = block_matching_flow(img_prev, img_next)

    err = np.sqrt(np.sum((flow_noc[:, :, :2]-flow)**2, axis=2))
    noc = flow_noc[:, :, 2].astype(bool)
    msen = np.mean(err[noc]**2)
    pepn = np.sum(err[noc] > 3) / err.size
    print(f'MSEN: {msen:.4f}, PEPN: {pepn:.4f}')

    cv2.imshow('flow', draw_flow(img_prev, flow))
    #cv2.imshow('flow', draw_hsv(flow))
    cv2.waitKey(0)
