import numpy as np
import cv2


def read_calibration(filename):
    with open(filename, 'r') as f:
        line = f.readline()
    line = line.split(': ')[1]

    homography = []
    for row in line.split(';'):
        homography.append([float(x) for x in row.split(' ')])

    return np.array(homography)


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


if __name__ == '__main__':
    import os
    from utils.aicity_reader import parse_annotations_from_txt, group_by_id

    root = '../../../data/AIC20_track3/train/S03/c014'
    detections = group_by_id(parse_annotations_from_txt(os.path.join(root, 'gt', 'gt.txt')))
    H = read_calibration(os.path.join(root, 'calibration.txt'))
    cap = cv2.VideoCapture(os.path.join(root, 'vdo.avi'))
    fps = cap.get(cv2.CAP_PROP_FPS)

    #id = np.random.choice(list(detections.keys()))
    id = 242

    track_3d = []
    for det in sorted(detections[id], key=lambda det: det.frame):
        u, v = (det.xtl + det.xbr) / 2, det.ybr  # bottom center
        lat, lon = image2world(u, v, H)  # backproject to obtain latitude/longitude in degrees
        lat, lon = degrees2meters(lat, lon)  # convert degrees to meters
        track_3d.append(np.array([lat, lon]))

        cap.set(cv2.CAP_PROP_POS_FRAMES, det.frame)
        ret, img = cap.read()
        if len(track_3d) >= 5:
            img = cv2.rectangle(img, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ybr)), (0, 255, 0), 2)
            speed = magnitude(estimate_speed(np.array(track_3d[-5:]), fps))
            img = cv2.putText(img, f'{speed*3.6:.2f} km/h', (int(det.xtl), int(det.ytl)-10), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0), 2)
        cv2.imshow('tracks', cv2.resize(img, (960, 540)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    track_3d = np.array(track_3d)

    speed = estimate_speed(track_3d, fps)
    print(f'id: {id}, avg speed: ({speed[0]*3.6:.2f}, {speed[1]*3.6:.2f}) km/h')
