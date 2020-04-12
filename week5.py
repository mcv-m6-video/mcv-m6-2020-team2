import os
import numpy as np
import cv2
import imageio
from tqdm import trange

from src.tracking.tracking import update_tracks_by_overlap, remove_static_tracks
from src.tracking.sort import Sort
from detection.detection import Detection
from src.evaluation.idf1 import MOTAcumulator
from src.utils.aicity_reader import AICityChallengeAnnotationReader, group_by_frame
from src.evaluation.average_precision import mean_average_precision


def task1(save_path='results/week5/task_1',
          distance_thresholds=[675],
          min_track_len=5,
          min_width=60,
          min_height=48,
          sequence='S03',
          camera='c010',
          detector='yolo3'): # 'mask_rcnn', 'ssd512', 'yolo3'

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
        detections = group_by_frame(detections)

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


def task1_optical_flow(save_path='results/week5/task_1_optical_flow',
                       distance_thresholds=[675],
                       min_track_len=5,
                       min_width=60,
                       min_height=48,
                       sequence='S03',
                       camera='c010',
                       detector='yolo3'): # 'mask_rcnn', 'ssd512', 'yolo3'

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
    previous_frame = None
    video_percentage = 1
    start = 0
    end = int(n_frames * video_percentage)

    for frame in trange(start, end, desc='Tracking'):

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, img = cap.read()

        detections_on_frame_ = dets.get(frame, [])
        detections_on_frame = []
        for d in detections_on_frame_:
            if min_width < (d.ybr - d.ytl) and min_height < (d.xbr - d.xtl):
                detections_on_frame.append(d)

        if frame == 0 or not detections_on_frame:
            optical_flow = None
        else:
            height, width = previous_frame.shape[:2]

            # get points on which to detect the flow
            points = []
            for det in detections_on_frame:
                points.append([det.xtl, det.ytl])
                points.append([det.xbr, det.ybr])
            p0 = np.array(points, dtype=np.float32)

            # params for lucas-kanade optical flow
            lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

            p1, st, err = cv2.calcOpticalFlowPyrLK(previous_frame, img, p0, None, **lk_params)

            p0 = p0.reshape((len(detections_on_frame)*2, 2))
            p1 = p1.reshape((len(detections_on_frame)*2, 2))
            st = st.reshape(len(detections_on_frame)*2)

            # flow field computed by subtracting prev points from next points
            flow = p1 - p0
            flow[st == 0] = 0

            optical_flow = np.zeros((height, width, 2), dtype=np.float32)
            for jj, det in enumerate(detections_on_frame):
                optical_flow[int(det.ytl), int(det.xtl)] = flow[2*jj]
                optical_flow[int(det.ybr), int(det.xbr)] = flow[2*jj+1]


        previous_frame = img.copy()

        tracks, frame_tracks, max_track = update_tracks_by_overlap(tracks,
                                                                   detections_on_frame,
                                                                   max_track,
                                                                   refinement=False, 
                                                                   optical_flow=optical_flow)

        y_true.append(gt.get(frame, []))

    idf1s = []
    for distance_threshold in distance_thresholds:
        accumulator = MOTAcumulator()
        y_pred = []

        moving_tracks = remove_static_tracks(tracks, distance_threshold, min_track_len)
        detections = []
        for track in moving_tracks:
            detections.extend(track.track)
        detections = group_by_frame(detections)

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


def task1_kalman_filter(save_path='results/week5/task_1_kalman_filter',
                        distance_thresholds=[675],
                        min_track_len=5,
                        min_width=60,
                        min_height=48,
                        sequence='S03',
                        camera='c010',
                        detector='yolo3'): # 'mask_rcnn', 'ssd512', 'yolo3'

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
    tracker = Sort()
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

        detections_on_frame = tracker.update(np.array([[*d.bbox, d.score] for d in detections_on_frame]))
        detections_on_frame = [Detection(frame, int(d[-1]), 'car', *d[:4]) for d in detections_on_frame]

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
        detections = group_by_frame(detections)

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

        ap, prec, rec = mean_average_precision(y_true, y_pred, classes=['car'], sort_method=None)
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
