import os
from collections import defaultdict

import numpy as np
import cv2

from utils.aicity_reader import parse_annotations_from_txt


def plot_tracks(annotations_file, video_file):
    detections = parse_annotations_from_txt(annotations_file)
    cap = cv2.VideoCapture(video_file)

    frame_detections = defaultdict(list)
    for det in detections:
        frame_detections[det.frame].append(det)

    for frame in frame_detections.keys():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, img = cap.read()
        for det in frame_detections[frame]:
            np.random.seed(det.id)
            color = tuple(np.random.randint(0, 256, 3).tolist())
            img = cv2.rectangle(img, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ybr)), color, 2)

            text = str(det.id)
            font = cv2.FONT_HERSHEY_COMPLEX
            scale = 0.75
            thickness = 2
            size = cv2.getTextSize(text, font, scale, thickness)
            img = cv2.putText(img, text, (int(det.xtl), int(det.ytl - size[1])), font, scale, color, thickness)

        cv2.imshow('tracks', cv2.resize(img, (960, 540)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    root = '../../../data/AIC20_track3/train/S03/c010'
    plot_tracks(
        annotations_file=os.path.join(root, 'mtsc', 'mtsc_tc_mask_rcnn.txt'),
        video_file=os.path.join(root, 'vdo.avi')
    )
