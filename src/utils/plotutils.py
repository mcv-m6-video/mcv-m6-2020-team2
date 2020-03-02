import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from src.evaluation.intersection_over_union import mean_intersection_over_union


def video_iou_plot(gt, det, video_path, title='', save_path=None):
    frames = sorted(list(set(gt) & set(det)))
    overlaps = []
    for frame in frames:
        boxes1 = [box[1:5] for box in gt[frame]]
        boxes2 = [box[1:5] for box in det[frame]]
        iou = mean_intersection_over_union(boxes1, boxes2)
        overlaps.append(iou)

    cap = cv2.VideoCapture(video_path)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    fig, ax = plt.subplots(2)
    image = ax[0].imshow(np.zeros((height, width)))
    line, = ax[1].plot(frames, overlaps)
    artists = [image, line]

    def update(i):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames[i])
        ret, img = cap.read()
        for box in gt[frames[i]]:
            cv2.rectangle(img, (int(box[1]), int(box[2])), (int(box[3]), int(box[4])), (0, 255, 0), 2)
        for box in det[frames[i]]:
            cv2.rectangle(img, (int(box[1]), int(box[2])), (int(box[3]), int(box[4])), (0, 0, 255), 2)
        artists[0].set_data(img[:, :, ::-1])
        artists[1].set_data(frames[:i + 1], overlaps[:i + 1])
        return artists

    ani = animation.FuncAnimation(fig, update, len(frames), interval=2, blit=True)
    if save_path is not None:
        ani.save(os.path.join(save_path, 'video_iou.gif'), writer='ffmpeg')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_ylim(0, 1)
    ax[1].set_xlabel('#frame')
    ax[1].set_ylabel('mean IoU')
    fig.suptitle(title)
    plt.show()
