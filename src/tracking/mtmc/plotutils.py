import os

import numpy as np
import cv2
import matplotlib.pyplot as plt

from utils.aicity_reader import parse_annotations_from_txt, group_by_frame, group_by_id
from tracking.mtmc.camera import read_calibration, read_timestamps, image2world, degrees2meters, estimate_speed, magnitude


def draw_detections(img, detections):
    for det in detections:
        np.random.seed(det.id)
        color = tuple(np.random.randint(0, 256, 3).tolist())
        img = cv2.rectangle(img, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ybr)), color, 2)
        text_params = {'fontFace': cv2.FONT_HERSHEY_DUPLEX, 'fontScale': 0.75, 'thickness': 2}
        img = cv2.putText(img, str(det.id), (int(det.xtl), int(det.ytl - 10)), color=color, **text_params)
        img = cv2.putText(img, str(det.frame), (10, 30), color=(255, 255, 255), **text_params)
    return img


def plot_tracks(root):
    detections = group_by_frame(parse_annotations_from_txt(os.path.join(root, 'mtsc', 'mtsc_tc_mask_rcnn.txt')))
    cap = cv2.VideoCapture(os.path.join(root, 'vdo.avi'))

    for frame in detections.keys():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, img = cap.read()

        img = draw_detections(img, detections[frame])

        cv2.imshow('tracks', cv2.resize(img, (960, 540)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def plot_speed(root, id):
    detections = group_by_id(parse_annotations_from_txt(os.path.join(root, 'gt', 'gt.txt')))
    H = read_calibration(os.path.join(root, 'calibration.txt'))
    cap = cv2.VideoCapture(os.path.join(root, 'vdo.avi'))
    fps = cap.get(cv2.CAP_PROP_FPS)

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
            img = cv2.putText(img, f'{speed * 3.6:.2f} km/h', (int(det.xtl), int(det.ytl) - 10),
                              cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0), 2)
        cv2.imshow('tracks', cv2.resize(img, (960, 540)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    track_3d = np.array(track_3d)

    speed = estimate_speed(track_3d, fps)
    print(f'id: {id}, avg speed: ({speed[0] * 3.6:.2f}, {speed[1] * 3.6:.2f}) km/h')


def plot_timeline(root, seq, id):
    timestamps = read_timestamps(os.path.join(root, 'cam_timestamp', f'{seq}.txt'))

    ranges = {}
    for cam in timestamps.keys():
        detections = group_by_id(parse_annotations_from_txt(os.path.join(root, 'train', seq, cam, 'gt', 'gt.txt')))
        if id in detections:
            cap = cv2.VideoCapture(os.path.join(root, 'train', seq, cam, 'vdo.avi'))
            fps = cap.get(cv2.CAP_PROP_FPS)
            id_detections = sorted(detections[id], key=lambda det: det.frame)
            start_time = timestamps[cam] + id_detections[0].frame / fps
            end_time = timestamps[cam] + id_detections[-1].frame / fps
            ranges[cam] = (start_time, end_time)

    plt.barh(range(len(ranges)), [end-start for start, end in ranges.values()], left=[start for start, _ in ranges.values()])
    plt.yticks(range(len(ranges)), list(ranges.keys()))
    plt.title(f'id = {id}')
    plt.show()


if __name__ == '__main__':
    # plot_tracks('../../../data/AIC20_track3/train/S03/c013')
    # plot_speed('../../../data/AIC20_track3/train/S03/c014', 242)
    plot_timeline('../../../data/AIC20_track3', 'S03', 242)
