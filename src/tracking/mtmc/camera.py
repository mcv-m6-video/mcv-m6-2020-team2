import numpy as np

CAMERA_LOCATION = {
    'c010': (42.497670, -90.673415),
    'c011': (42.496823, -90.673974),
    'c012': (42.496574, -90.674223),
    'c013': (42.496411, -90.674445),
    'c014': (42.495918, -90.674981),
    'c015': (42.496476, -90.675680)
}


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


def project_bbox(bbox, H1, H2):
    xtl, ytl, xbr, ybr = bbox
    bl = world2image(*image2world(xtl, ybr, H1), H2)
    br = world2image(*image2world(xbr, ybr, H1), H2)
    new_width = br[0]-bl[0]
    new_height = new_width / (xbr-xtl) * (ybr-ytl)
    new_bbox = (bl[0], bl[1]-new_height, br[0], br[1])
    return new_bbox


def bbox2gps(bbox, H):
    xtl, ytl, xbr, ybr = bbox
    xc = (xtl + xbr) / 2
    lat, lon = image2world(xc, ybr, H)
    return np.array([lat, lon])


def degrees2meters(lat, lon):
    # https://en.wikipedia.org/wiki/Geographic_coordinate_system#Length_of_a_degree
    lat = 111132.92 - 559.82 * np.cos(2*lat) + 1.175 * np.cos(4*lat) - 0.0023 * np.cos(6*lat)
    lon = 111412.84 * np.cos(lon) - 93.5 * np.cos(3*lon) + 0.118 * np.cos(5*lon)
    return lat, lon


def magnitude(x):
    return np.sqrt(np.sum(x**2))


def angle(x, y):
    return np.rad2deg(np.arccos(np.sum(x * y) / (magnitude(x) * magnitude(y))))


def angle_to_cam(track, H, cam, num_frames=10):
    """Track should be sorted by frame."""
    # track.sort(key=lambda det: det.frame)

    num_frames = min(num_frames, len(track))
    speed_dir = bbox2gps(track[-1].bbox, H) - bbox2gps(track[-num_frames].bbox, H)
    cam_dir = np.array(CAMERA_LOCATION[cam]) - bbox2gps(track[-num_frames].bbox, H)

    return angle(speed_dir, cam_dir)


def time_range(track, timestamp, fps):
    """Track should be sorted by frame."""
    # track.sort(key=lambda det: det.frame)

    start_time = timestamp + track[0].frame / fps
    end_time = timestamp + track[-1].frame / fps

    return start_time, end_time


if __name__ == '__main__':
    import os
    from utils.aicity_reader import parse_annotations_from_txt, group_by_id

    cam = 'c012'
    root = os.path.join('../../../data/AIC20_track3/train/S03', cam)
    detections = group_by_id(parse_annotations_from_txt(os.path.join(root, 'gt', 'gt.txt')))
    H = read_calibration(os.path.join(root, 'calibration.txt'))

    id = np.random.choice(list(detections.keys()))
    track = sorted(detections[id], key=lambda det: det.frame)
    for c in [f'c{c:03d}' for c in range(10, 16)]:
        if c != cam:
            a = angle_to_cam(track, H, c)
            print(f'{c}, {id}, {a:.2f}', 'going' if a < 90 else 'coming')
