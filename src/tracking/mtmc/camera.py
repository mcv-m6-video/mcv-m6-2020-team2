import numpy as np


def read_calibration(filename):
    with open(filename, 'r') as f:
        line = f.readline()
    line = line.split(': ')[1]

    homography = []
    for row in line.split(';'):
        homography.append([float(x) for x in row.split(' ')])

    return np.array(homography)


def read_timestamps(filename):
    timestamps = {}
    with open(filename, 'r') as f:
        for line in f:
            items = line.split(' ')
            cam = items[0]
            timestamp = float(items[1])
            timestamps[cam] = timestamp
    return timestamps


def image2world(u, v, H):
    Xi_h = np.array([u, v, 1])
    Xw_h = np.linalg.inv(H).dot(Xi_h)
    lat, lon = Xw_h[:2] / Xw_h[2]
    return lat, lon


def world2image(lat, lon, H):
    Xw_h = np.array([lat, lon, 1])
    Xi_h = H.dot(Xw_h)
    u, v = Xi_h[:2] / Xi_h[2]
    return u, v


def warp_bbox(bbox, H1, H2):
    xtl, ytl, xbr, ybr = bbox
    bl = world2image(*image2world(xtl, ybr, H1), H2)
    br = world2image(*image2world(xbr, ybr, H1), H2)
    new_width = br[0]-bl[0]
    new_height = new_width / (xbr-xtl) * (ybr-ytl)
    new_bbox = (bl[0], bl[1]-new_height, br[0], br[1])
    return new_bbox


def degrees2meters(lat, lon):
    # https://en.wikipedia.org/wiki/Geographic_coordinate_system#Length_of_a_degree
    lat = 111132.92 - 559.82 * np.cos(2*lat) + 1.175 * np.cos(4*lat) - 0.0023 * np.cos(6*lat)
    lon = 111412.84 * np.cos(lon) - 93.5 * np.cos(3*lon) + 0.118 * np.cos(5*lon)
    return lat, lon


def estimate_speed(track, fps, w=10):
    w = min(w, len(track)-1)
    return np.mean(np.abs(track[w:]-track[:-w]), axis=0) * fps / w


def magnitude(x):
    return np.sqrt(np.sum(x**2))
