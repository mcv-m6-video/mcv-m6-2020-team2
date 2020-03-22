import time

import numpy as np
import cv2
from skimage.feature import match_template
from tqdm import trange


def distance(x1: np.ndarray, x2: np.ndarray, metric='euclidean'):
    if metric == 'euclidean':
        return np.sqrt(np.sum((x1 - x2) ** 2))
    elif metric == 'sad':
        return np.sum(np.abs(x1 - x2))
    elif metric == 'mad':
        return np.mean(np.abs(x1 - x2))
    elif metric == 'ssd':
        return np.sum((x1 - x2) ** 2)
    elif metric == 'mse':
        return np.mean((x1 - x2) ** 2)
    else:
        raise ValueError(f'Unknown distance metric: {metric}')


def block_matching(reference, target, metric='euclidean', algorithm='es'):
    """
    Search reference in target and return the position with maximum correlation
    """
    if algorithm == 'es':
        # exhaustive search
        min_dist = np.inf
        pos = (0, 0)
        for i in range(target.shape[0]-reference.shape[0]):
            for j in range(target.shape[1]-reference.shape[1]):
                dist = distance(reference, target[i:i+reference.shape[0], j:j+reference.shape[1]], metric)
                if dist < min_dist:
                    pos = (i, j)
                    min_dist = dist
        return pos
    elif algorithm == 'tss':
        # three step search
        step = 8
        orig = ((target.shape[0]-reference.shape[0])//2, (target.shape[1]-reference.shape[1])//2)
        step = (min(step, orig[0]), min(step, orig[1]))
        while step[0] > 1 and step[1] > 1:
            min_dist = np.inf
            pos = orig
            for i in [orig[0]-step[0], orig[0], orig[0]+step[0]]:
                for j in [orig[1]-step[1], orig[1], orig[1]+step[1]]:
                    dist = distance(reference, target[i:i + reference.shape[0], j:j + reference.shape[1]], metric)
                    if dist < min_dist:
                        pos = (i, j)
                        min_dist = dist
            orig = pos
            step = (step[0]//2, step[1]//2)
        return orig
    elif algorithm == 'corr':
        # normalized correlation
        corr = match_template(target, reference)
        pos = np.unravel_index(np.argmax(corr), corr.shape)
        return pos
    else:
        raise ValueError(f'Unknown block matching algorithm: {algorithm}')


def block_matching_flow(img_prev: np.ndarray, img_next: np.ndarray, block_size=16, search_area=16,
                        motion_type='backward', metric='euclidean', algorithm='es'):
    """
    Compute block-matching based motion estimation
    """
    if motion_type == 'forward':
        reference = img_prev
        target = img_next
    elif motion_type == 'backward':
        reference = img_next
        target = img_prev
    else:
        raise ValueError(f'Unknown motion type: {motion_type}')

    assert (reference.shape == target.shape)
    height, width = reference.shape[:2]

    flow_field = np.zeros((height, width, 2), dtype=float)
    for i in trange(0, height - block_size, block_size):
        for j in range(0, width - block_size, block_size):
            ii = max(i-search_area, 0)
            jj = max(j-search_area, 0)
            disp = block_matching(reference[i:i+block_size, j:j+block_size],
                                  target[ii: min(i+block_size+search_area, height),
                                         jj: min(j+block_size+search_area, width)],
                                  metric, algorithm)
            u = disp[1] - (j - jj)
            v = disp[0] - (i - ii)
            flow_field[i:i + block_size, j:j + block_size, :] = [u, v]

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


def draw_hsv(flow, scale=4):
    h, w = flow.shape[:2]
    fx, fy = flow[..., 0], flow[..., 1]
    ang = np.arctan2(fy, fx) + np.pi
    mag = np.sqrt(fx * fx + fy * fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(mag * scale, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def read_flow(filename):
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    fx = (img[..., 2].astype(float) - 2 ** 15) / 64
    fy = (img[..., 1].astype(float) - 2 ** 15) / 64
    f_valid = img[..., 0].astype(bool)
    fx[~f_valid] = 0
    fy[~f_valid] = 0
    flow = np.dstack((fx, fy, f_valid))
    return flow


if __name__ == '__main__':
    img_prev = cv2.imread('../../data/data_stereo_flow/training/image_0/000045_10.png', cv2.IMREAD_GRAYSCALE)
    img_next = cv2.imread('../../data/data_stereo_flow/training/image_0/000045_11.png', cv2.IMREAD_GRAYSCALE)
    flow_noc = read_flow('../../data/data_stereo_flow/training/flow_noc/000045_10.png')

    tic = time.time()
    flow = block_matching_flow(img_prev, img_next, algorithm='corr')
    toc = time.time()
    print(f'runtime: {toc-tic:.3f}s')

    err = np.sqrt(np.sum((flow_noc[..., :2] - flow) ** 2, axis=2))
    noc = flow_noc[..., 2].astype(bool)
    msen = np.mean(err[noc] ** 2)
    pepn = np.sum(err[noc] > 3) / err[noc].size
    print(f'MSEN: {msen:.4f}, PEPN: {pepn:.4f}')

    cv2.imshow('flow', draw_flow(img_prev, flow))
    cv2.imshow('hsv', draw_hsv(flow, scale=16))
    cv2.waitKey(0)
