import os
import numpy as np
import cv2
import imageio
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

from src.tracking.tracking import update_tracks_by_overlap, remove_static_tracks
from src.evaluation.idf1 import MOTAcumulator
from src.utils.aicity_reader import AICityChallengeAnnotationReader, group_detections_by_frame
from src.evaluation.average_precision import mean_average_precision


def task1(save_path='results/week5/task_1',
          distance_thresholds=[550],
          min_track_len=5,
          min_width=60,
          min_height=48,
          sequence='S03',
          camera='c010',
          detector='mask_rcnn'): # 'mask_rcnn', 'ssd512', 'yolo3'

    save_video = False
    save_summary = False
    os.makedirs(save_path, exist_ok=True)

    reader = AICityChallengeAnnotationReader(path='data/AICity_data/train/' + sequence + '/' + camera + '/gt/gt.txt')
    gt = reader.get_annotations(classes=['car'])
    reader = AICityChallengeAnnotationReader(path='data/AICity_data/train/' + sequence + '/' + camera + '/det/det_' + detector + '.txt')
    dets = reader.get_annotations(classes=['car'])

    cap = cv2.VideoCapture('data/AICity_data/train/' + sequence + '/' + camera + '/vdo.avi')
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if save_video:
        writer = imageio.get_writer(os.path.join(save_path, 'task1_' + sequence + '_' + camera + '_' + detector + '.gif'), fps=fps)

    accumulator = MOTAcumulator()
    y_true = []
    y_pred = []
    tracks = []
    max_track = 0
    video_percentage = 1
    start = 0
    end = int(n_frames * video_percentage)

    for frame in trange(start, end, desc='Tracking'):

        detections_on_frame_ = dets.get(frame, [])
        detections_on_frame = []
        for d in detections_on_frame_:
            if min_width < (d.ybr - d.ytl) and min_height < (d.xbr - d.xtl):
                detections_on_frame.append(d)

        tracks, frame_tracks, max_track = update_tracks_by_overlap(tracks,
                                                                   detections_on_frame,
                                                                   max_track,
                                                                   refinement=False, 
                                                                   optical_flow=None)

        y_true.append(gt.get(frame, []))

    idf1s = []
    for distance_threshold in distance_thresholds:
        accumulator = MOTAcumulator()
        y_pred = []

        moving_tracks = remove_static_tracks(tracks, distance_threshold, min_track_len)
        detections = []
        for track in moving_tracks:
            detections.extend(track.track)
        detections = group_detections_by_frame(detections)

        for frame in trange(start, end, desc='Accumulating detections'):

            if save_video:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ret, img = cap.read()

                for det in y_true[frame]:
                    cv2.rectangle(img, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ybr)), (0, 255, 0), 6)

            frame_detections = []
            for det in detections.get(frame, []):
                frame_detections.append(det)
                if save_video:
                    cv2.rectangle(img, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ybr)), track.color, 6)
                    cv2.rectangle(img, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ytl) - 15), track.color, -6)
                    cv2.putText(img, str(det.id), (int(det.xtl), int(det.ytl)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 6)
                    cv2.circle(img, track.track[-1].center, 5, track.color, -1)

            y_pred.append(frame_detections)

            if save_video:
                writer.append_data(cv2.resize(img, (600, 350)))

            accumulator.update(y_true[frame], y_pred[-1])

        ap, prec, rec = mean_average_precision(y_true, y_pred, classes=['car'], sort_method='score')
        print(f'AP: {ap:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}')
        print('Additional metrics:')
        summary = accumulator.get_idf1()
        # As mentioned in https://github.com/cheind/py-motmetrics:
        #     FAR = FalsePos / Frames * 100
        #     MOTP = (1 - MOTP) * 100
        print(summary)

        if save_summary:
            with open(os.path.join(save_path, 'task1_' + sequence + '_' + camera + '_' + detector + '_' + str(distance_threshold) + '.txt'), 'w') as f:
                f.write(str(summary))

        idf1s.append(summary['idf1']['acc']*100)

    cv2.destroyAllWindows()
    if save_video:
        writer.close()

    print()

    return idf1s


if __name__ == '__main__':
    task1()
