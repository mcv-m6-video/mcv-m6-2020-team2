import os

import numpy as np
import cv2
import matplotlib.pyplot as plt

from utils.aicity_reader import parse_annotations_from_txt, group_by_frame


def plot_tracks(annotations_file, video_file):
    detections = group_by_frame(parse_annotations_from_txt(annotations_file))
    cap = cv2.VideoCapture(video_file)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame in detections.keys():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, img = cap.read()
        for det in detections[frame]:
            np.random.seed(det.id)
            color = tuple(np.random.randint(0, 256, 3).tolist())

            img = cv2.rectangle(img, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ybr)), color, 2)

            text_params = {'fontFace': cv2.FONT_HERSHEY_DUPLEX, 'fontScale': 0.75, 'thickness': 2}
            img = cv2.putText(img, str(det.id), (int(det.xtl), int(det.ytl-10)), color=color, **text_params)

            img = cv2.putText(img, f'{frame}/{length}', (10, 30), color=(255, 255, 255), **text_params)

        cv2.imshow('tracks', cv2.resize(img, (960, 540)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def plot_timeline(root, seq):
    timestamps_file = os.path.join(root, 'cam_timestamp', f'{seq}.txt')
    timestamps = {}
    with open(timestamps_file, 'r') as f:
        for line in f:
            items = line.split(' ')
            cam = items[0]
            timestamp = float(items[1])
            timestamps[cam] = timestamp

    ranges = {}
    for cam in timestamps.keys():
        cap = cv2.VideoCapture(os.path.join(root, 'train', seq, cam, 'vdo.avi'))
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_time = timestamps[cam]
        end_time = start_time + length / fps
        ranges[cam] = (start_time, end_time)

    plt.barh(range(len(ranges)), [end-start for start, end in ranges.values()], left=[start for start, _ in ranges.values()])
    plt.yticks(range(len(ranges)), list(ranges.keys()))
    plt.show()


if __name__ == '__main__':
    root = '../../../data/AIC20_track3/train/S03/c014'
    plot_tracks(
        annotations_file=os.path.join(root, 'gt', 'gt.txt'),
        video_file=os.path.join(root, 'vdo.avi')
    )

    # plot_timeline('../../../data/AIC20_track3', 'S03')
