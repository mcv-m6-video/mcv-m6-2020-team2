import os
from glob import glob

import numpy as np
import cv2
import imageio
from tqdm import trange
from matplotlib import pyplot as plt

from src.tracking.tracking import update_tracks_by_overlap, remove_static_tracks
from src.tracking.mtmc.reid import reid, write_results
from src.tracking.sort import Sort
from detection.detection import Detection
from src.evaluation.idf1 import MOTAcumulator
from src.utils.aicity_reader import AICityChallengeAnnotationReader, group_by_frame, parse_annotations_from_txt
from src.evaluation.average_precision import mean_average_precision


def task1():
    test_type = 'one_detector_all_cameras'  # ['distance_thresholds', 'min_width_length', 'mean_idf1_across_cameras_sequence_03']

    # TEST DISTANCE THRESHOLDS ON DETECTORS
    if test_type == 'distance_thresholds':

        save_path = 'results/week5/task_1'
        distance_thresholds = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450,
                               475, 500, 525, 550, 575, 600, 625, 650, 675, 700, 725, 750, 775, 800, 825, 850, 875, 900,
                               925, 950, 975, 1000]
        min_track_len = 5
        min_width = 0
        min_height = 0
        sequence = 'S03'
        camera = 'c010'
        detectors = ['mask_rcnn', 'ssd512', 'yolo3']

        all_idf1s = []
        for detector in detectors:
            idf1s = launch_test(
                save_path=save_path,
                distance_thresholds=distance_thresholds,
                min_track_len=min_track_len,
                min_width=min_width,
                min_height=min_height,
                sequence=sequence,
                camera=camera,
                detector=detector)

            all_idf1s.append(idf1s)

        for idf1s in all_idf1s:
            plt.plot(distance_thresholds, idf1s)
        plt.xticks([d for d in distance_thresholds if d % 100 == 0])
        plt.xlabel('Distance thresholds')
        plt.ylabel('IDF1')
        plt.legend(detectors, loc='best')
        if save_path:
            plt.savefig(os.path.join(save_path, 'task1_' + sequence + '_' + camera + '_' + 'dist-th_vs_idf1.png'))
        plt.show()

    elif test_type == 'min_width_length':

        save_path = 'results/week5/task_1'
        distance_thresholds = [550]
        min_track_len = 5
        min_widths = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]
        min_heights = [int(0.8 * w) for w in min_widths]
        sequence = 'S03'
        camera = 'c010'
        detectors = ['mask_rcnn', 'ssd512', 'yolo3']

        all_idf1s = []
        for detector in detectors:
            idf1s_detector = []
            for min_width, min_height in zip(min_widths, min_heights):
                idf1s = launch_test(
                    save_path=save_path,
                    distance_thresholds=distance_thresholds,
                    min_track_len=min_track_len,
                    min_width=min_width,
                    min_height=min_height,
                    sequence=sequence,
                    camera=camera,
                    detector=detector)

                idf1s_detector.append(idf1s)

            all_idf1s.append(idf1s_detector)

        for idf1s in all_idf1s:
            plt.plot(min_widths, idf1s)
        plt.xticks([w for w in min_widths if w % 40 == 0])
        plt.xlabel('Minimum width (length = 0.8 * width)')
        plt.ylabel('IDF1')
        plt.legend(detectors, loc='best')
        if save_path:
            plt.savefig(
                os.path.join(save_path, 'task1_' + sequence + '_' + camera + '_' + 'min-width-length_vs_idf1.png'))
        plt.show()

    elif test_type == 'one_detector_all_cameras':

        ## No optical flow
        save_path = 'results/week5/task_1'
        distance_thresholds = [675]
        min_track_len = 5
        min_width = 60
        min_height = 48
        sequence = 'S03'
        cameras = [os.path.split(p)[-1] for p in glob('data/AICity_data/train/' + sequence + '/c*')]
        detector = 'yolo3'

        all_idf1s = []
        for camera in cameras:
            idf1s = launch_test(
                save_path=save_path,
                distance_thresholds=distance_thresholds,
                min_track_len=min_track_len,
                min_width=min_width,
                min_height=min_height,
                sequence=sequence,
                camera=camera,
                detector=detector)

            all_idf1s.append(idf1s[0])

        print(f'IDF1s for detector {detector} (mean IDF1 {np.mean(np.array(all_idf1s))}):')
        for idf1s, camera in zip(all_idf1s, cameras):
            print(f'\tcamera {camera}: {idf1s}')

        ## Optical flow
        # save_path = 'results/week5/task_1_optical_flow'
        # distance_thresholds = [675]
        # min_track_len = 5
        # min_width = 60
        # min_height = 48
        # sequence = 'S04'
        # cameras = [os.path.split(p)[-1] for p in glob('data/AICity_data/train/' + sequence + '/c*')]
        # detector = 'yolo3'

        # all_idf1s = []
        # for camera in cameras:
        #     idf1s = launch_test_optical_flow(
        #         save_path=save_path,
        #         distance_thresholds=distance_thresholds,
        #         min_track_len=min_track_len,
        #         min_width=min_width,
        #         min_height=min_height,
        #         sequence=sequence,
        #         camera=camera,
        #         detector=detector)

        #     all_idf1s.append(idf1s[0])

        # print(f'IDF1s for detector {detector} (mean IDF1 {np.mean( np.array(all_idf1s))}):')
        # for idf1s, camera in zip(all_idf1s, cameras):
        #     print(f'\tcamera {camera}: {idf1s}')

        ## Kalman filter
        # save_path = 'results/week5/task_1_kalman_filter'
        # distance_thresholds = [675]
        # min_track_len = 5
        # min_width = 60
        # min_height = 48
        # sequence = 'S04'
        # cameras = [os.path.split(p)[-1] for p in glob('data/AICity_data/train/' + sequence + '/c*')]
        # detector = 'mask_rcnn'

        # all_idf1s = []
        # for camera in cameras:
        #     idf1s = launch_test_kalman_filter(
        #         save_path=save_path,
        #         distance_thresholds=distance_thresholds,
        #         min_track_len=min_track_len,
        #         min_width=min_width,
        #         min_height=min_height,
        #         sequence=sequence,
        #         camera=camera,
        #         detector=detector)

        #     all_idf1s.append(idf1s[0])

        # print(f'IDF1s for detector {detector} (mean IDF1 {np.mean( np.array(all_idf1s))}):')
        # for idf1s, camera in zip(all_idf1s, cameras):
        #     print(f'\tcamera {camera}: {idf1s}')

    elif test_type == 'mean_idf1_across_cameras_sequence_03':

        save_path = 'results/week5/task_1'
        distance_thresholds = [400, 425, 450, 475, 500, 525, 550, 575, 600, 625, 650, 675, 700, 725, 750, 775, 800]
        min_track_len = 5
        min_width = 60
        min_height = 48
        sequence = 'S03'
        cameras = [os.path.split(p)[-1] for p in glob('data/AICity_data/train/' + sequence + '/c*')]
        detectors = ['mask_rcnn', 'ssd512', 'yolo3']

        all_idf1s = []
        for detector in detectors:
            idf1s_detector = []
            for camera in cameras:
                idf1s = launch_test(
                    save_path=save_path,
                    distance_thresholds=distance_thresholds,
                    min_track_len=min_track_len,
                    min_width=min_width,
                    min_height=min_height,
                    sequence=sequence,
                    camera=camera,
                    detector=detector)

                idf1s_detector.append(idf1s)

            idf1s_detector = list(np.mean(np.array(idf1s_detector), axis=0))

            all_idf1s.append(idf1s_detector)

        for idf1s, detector in zip(all_idf1s, detectors):
            plt.plot(distance_thresholds, idf1s)
        plt.xticks([d for d in distance_thresholds if d % 50 == 0])
        plt.xlabel('distance_thresholds')
        plt.ylabel('Mean IDF1 across cameras')
        plt.legend(detectors, loc='best')
        if save_path:
            plt.savefig(os.path.join(save_path, 'task1_mean-idf1-across-cameras-sequence-03.png'))
        plt.show()


def launch_test(save_path, distance_thresholds, min_track_len, min_width, min_height, sequence, camera, detector):
    save_video = False
    save_summary = False
    save_tracks_txt = True
    fps = 24
    os.makedirs(save_path, exist_ok=True)

    reader = AICityChallengeAnnotationReader(path='data/AICity_data/train/' + sequence + '/' + camera + '/gt/gt.txt')
    gt = reader.get_annotations(classes=['car'])
    reader = AICityChallengeAnnotationReader(
        path='data/AICity_data/train/' + sequence + '/' + camera + '/det/det_' + detector + '.txt')
    dets = reader.get_annotations(classes=['car'])

    cap = cv2.VideoCapture('data/AICity_data/train/' + sequence + '/' + camera + '/vdo.avi')
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if save_video:
        writer = imageio.get_writer(
            os.path.join(save_path, 'task1_' + sequence + '_' + camera + '_' + detector + '.gif'), fps=fps)

    y_true = []
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
    all_moving_tracks = []
    for distance_threshold in distance_thresholds:
        accumulator = MOTAcumulator()
        y_pred = []

        moving_tracks = remove_static_tracks(tracks, distance_threshold, min_track_len)
        all_moving_tracks.extend(moving_tracks)
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
                    cv2.putText(img, str(det.id), (int(det.xtl), int(det.ytl)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0),
                                6)
                    cv2.circle(img, track.track[-1].center, 5, track.color, -1)

            y_pred.append(frame_detections)

            if save_video:
                writer.append_data(cv2.resize(img, (600, 350)))

            accumulator.update(y_true[frame], y_pred[-1])

        # Save tracks on .txt
        if save_tracks_txt:
            filename = os.path.join(save_path, sequence + '_' + camera + '.txt')

            lines = []
            for track in all_moving_tracks:
                for det in track.track:
                    lines.append(
                        (det.frame, track.id, det.xtl, det.ytl, det.width, det.height, det.score, "-1", "-1", "-1"))

            lines = sorted(lines, key=lambda x: x[0])
            with open(filename, "w") as file:
                for line in lines:
                    file.write(",".join(list(map(str, line))) + "\n")

        ap, prec, rec = mean_average_precision(y_true, y_pred, classes=['car'], sort_method='score')
        print(f'AP: {ap:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}')
        print('Additional metrics:')
        summary = accumulator.get_idf1()
        # As mentioned in https://github.com/cheind/py-motmetrics:
        #     FAR = FalsePos / Frames * 100
        #     MOTP = (1 - MOTP) * 100
        print(summary)

        if save_summary:
            with open(os.path.join(save_path, 'task1_' + sequence + '_' + camera + '_' + detector + '_' + str(
                    distance_threshold) + '.txt'), 'w') as f:
                f.write(str(summary))

        idf1s.append(summary['idf1']['acc'] * 100)

    cv2.destroyAllWindows()
    if save_video:
        writer.close()

    return idf1s


def launch_test_optical_flow(save_path, distance_thresholds, min_track_len, min_width, min_height, sequence, camera,
                             detector):
    save_video = False
    save_summary = False
    fps = 24
    os.makedirs(save_path, exist_ok=True)

    reader = AICityChallengeAnnotationReader(path='data/AICity_data/train/' + sequence + '/' + camera + '/gt/gt.txt')
    gt = reader.get_annotations(classes=['car'])
    reader = AICityChallengeAnnotationReader(
        path='data/AICity_data/train/' + sequence + '/' + camera + '/det/det_' + detector + '.txt')
    dets = reader.get_annotations(classes=['car'])

    cap = cv2.VideoCapture('data/AICity_data/train/' + sequence + '/' + camera + '/vdo.avi')
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if save_video:
        writer = imageio.get_writer(
            os.path.join(save_path, 'task1_' + sequence + '_' + camera + '_' + detector + '.gif'), fps=fps)

    y_true = []
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
            lk_params = dict(winSize=(15, 15), maxLevel=2,
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

            p1, st, err = cv2.calcOpticalFlowPyrLK(previous_frame, img, p0, None, **lk_params)

            p0 = p0.reshape((len(detections_on_frame) * 2, 2))
            p1 = p1.reshape((len(detections_on_frame) * 2, 2))
            st = st.reshape(len(detections_on_frame) * 2)

            # flow field computed by subtracting prev points from next points
            flow = p1 - p0
            flow[st == 0] = 0

            optical_flow = np.zeros((height, width, 2), dtype=np.float32)
            for jj, det in enumerate(detections_on_frame):
                optical_flow[int(det.ytl), int(det.xtl)] = flow[2 * jj]
                optical_flow[int(det.ybr), int(det.xbr)] = flow[2 * jj + 1]

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
                    cv2.putText(img, str(det.id), (int(det.xtl), int(det.ytl)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0),
                                6)
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
            with open(os.path.join(save_path, 'task1_' + sequence + '_' + camera + '_' + detector + '_' + str(
                    distance_threshold) + '.txt'), 'w') as f:
                f.write(str(summary))

        idf1s.append(summary['idf1']['acc'] * 100)

    cv2.destroyAllWindows()
    if save_video:
        writer.close()

    return idf1s


def launch_test_kalman_filter(save_path, distance_thresholds, min_track_len, min_width, min_height, sequence, camera,
                              detector):
    save_video = False
    save_summary = False
    fps = 24
    os.makedirs(save_path, exist_ok=True)

    reader = AICityChallengeAnnotationReader(path='data/AICity_data/train/' + sequence + '/' + camera + '/gt/gt.txt')
    gt = reader.get_annotations(classes=['car'])
    reader = AICityChallengeAnnotationReader(
        path='data/AICity_data/train/' + sequence + '/' + camera + '/det/det_' + detector + '.txt')
    dets = reader.get_annotations(classes=['car'])

    cap = cv2.VideoCapture('data/AICity_data/train/' + sequence + '/' + camera + '/vdo.avi')
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if save_video:
        writer = imageio.get_writer(
            os.path.join(save_path, 'task1_' + sequence + '_' + camera + '_' + detector + '.gif'), fps=fps)

    tracker = Sort()
    y_true = []
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
                    cv2.putText(img, str(det.id), (int(det.xtl), int(det.ytl)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0),
                                6)
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
            with open(os.path.join(save_path, 'task1_' + sequence + '_' + camera + '_' + detector + '_' + str(
                    distance_threshold) + '.txt'), 'w') as f:
                f.write(str(summary))

        idf1s.append(summary['idf1']['acc'] * 100)

    cv2.destroyAllWindows()
    if save_video:
        writer.close()

    return idf1s


def task2():
    root = 'data/AIC20_track3'
    seq = 'S03'
    model_path = 'src/tracking/metric_learning/checkpoints/epoch_19__ckpt.pth'
    reid_method = 'spatiotemporal'  # ['exhaustive', 'spatiotemporal', 'graph']

    # obtain reid results
    path_results = os.path.join('results', 'week5', seq)
    results = reid(root, seq, model_path, reid_method)
    write_results(results, path=path_results)

    # compute metrics
    accumulator = MOTAcumulator()
    for cam in os.listdir(os.path.join(root, 'train', seq)):
        dets_true = group_by_frame(parse_annotations_from_txt(os.path.join(root, 'train', seq, cam, 'gt', 'gt.txt')))
        dets_pred = group_by_frame(parse_annotations_from_txt(os.path.join(path_results, cam, 'results.txt')))
        for frame in dets_true.keys():
            y_true = dets_true.get(frame, [])
            y_pred = dets_pred.get(frame, [])
            accumulator.update(y_true, y_pred)
    print(f'Metrics: {accumulator.get_metrics()}')


if __name__ == '__main__':
    task1()
    #task2()
