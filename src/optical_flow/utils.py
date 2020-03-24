import numpy as np
import cv2


def read_flow(filename):
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    fx = (img[..., 2].astype(float) - 2 ** 15) / 64
    fy = (img[..., 1].astype(float) - 2 ** 15) / 64
    f_valid = img[..., 0].astype(bool)
    fx[~f_valid] = 0
    fy[~f_valid] = 0
    flow = np.dstack((fx, fy, f_valid))
    return flow


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
    # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 2] = np.minimum(mag * scale, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def evaluate_flow(flow_noc, flow):
    err = np.sqrt(np.sum((flow_noc[..., :2] - flow) ** 2, axis=2))
    noc = flow_noc[..., 2].astype(bool)
    msen = np.mean(err[noc] ** 2)
    pepn = np.sum(err[noc] > 3) / err[noc].size
    return msen, pepn
