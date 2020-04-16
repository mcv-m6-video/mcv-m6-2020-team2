import os
import time
from itertools import product
import imageio
from tqdm import tqdm

import numpy as np
import cv2
import pandas as pd

from src.optical_flow.block_matching_flow import block_matching_flow
from src.optical_flow.utils import read_flow, evaluate_flow, draw_flow, draw_hsv
from src.optical_flow.pyflow import pyflow

from src.video_stabilization.block_matching_stabilization import block_matching_stabilization
from src.video_stabilization.mesh_flow.stabilization import mesh_flow_main
from src.video_stabilization.point_feature_matching import point_feature_matching

from src.utils.aicity_reader import AICityChallengeAnnotationReader
from src.tracking.tracking import update_tracks_by_overlap
from src.evaluation.average_precision import mean_average_precision
from src.evaluation.idf1 import MOTAcumulator


def task1_1():
    # Optical Flow with Block Matching

    img_prev = cv2.imread('data/data_stereo_flow/training/image_0/000045_10.png', cv2.IMREAD_GRAYSCALE)
    img_next = cv2.imread('data/data_stereo_flow/training/image_0/000045_11.png', cv2.IMREAD_GRAYSCALE)
    flow_noc = read_flow('data/data_stereo_flow/training/flow_noc/000045_10.png')

    motion_type = ['forward', 'backward']
    search_area = [16, 32, 64, 128]
    block_size = [4, 8, 16, 32]

    data = []
    for m, p, n in product(motion_type, search_area, block_size):
        tic = time.time()
        flow = block_matching_flow(img_prev, img_next, motion_type=m, search_area=p, block_size=n, algorithm='corr')
        toc = time.time()
        msen, pepn = evaluate_flow(flow_noc, flow)
        data.append([m, p, n, msen, pepn, toc - tic])
    df = pd.DataFrame(data, columns=['motion_type', 'search_area', 'block_size', 'msen', 'pepn', 'runtime'])
    print(df)


def task1_2(algorithm='pyflow'):
    # Off-the-shelf Optical Flow

    img_prev = cv2.imread('data/data_stereo_flow/training/image_0/000045_10.png', cv2.IMREAD_GRAYSCALE)
    img_next = cv2.imread('data/data_stereo_flow/training/image_0/000045_11.png', cv2.IMREAD_GRAYSCALE)
    flow_noc = read_flow('data/data_stereo_flow/training/flow_noc/000045_10.png')

    if algorithm == 'pyflow':
        im1 = np.atleast_3d(img_prev.astype(float) / 255.)
        im2 = np.atleast_3d(img_next.astype(float) / 255.)

        # flow options:
        alpha = 0.012
        ratio = 0.75
        minWidth = 20
        nOuterFPIterations = 7
        nInnerFPIterations = 1
        nSORIterations = 30
        colType = 1  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

        tic = time.time()
        u, v, im2W = pyflow.coarse2fine_flow(im1, im2, alpha, ratio, minWidth, nOuterFPIterations,
                                             nInnerFPIterations, nSORIterations, colType)
        toc = time.time()
        flow = np.dstack((u, v))
    elif algorithm == 'lk':
        height, width = img_prev.shape[:2]

        # dense flow: one point for each pixel
        p0 = np.array([[x, y] for y in range(height) for x in range(width)], dtype=np.float32).reshape((-1, 1, 2))

        # params for lucas-kanade optical flow
        lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        tic = time.time()
        p1, st, err = cv2.calcOpticalFlowPyrLK(img_prev, img_next, p0, None, **lk_params)
        toc = time.time()

        p0 = p0.reshape((height, width, 2))
        p1 = p1.reshape((height, width, 2))
        st = st.reshape((height, width))

        # flow field computed by subtracting prev points from next points
        flow = p1 - p0
        flow[st == 0] = 0
    elif algorithm == 'fb':
        tic = time.time()
        flow = cv2.calcOpticalFlowFarneback(img_prev, img_next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        toc = time.time()
    else:
        raise ValueError(f'Unknown optical flow algorithm: {algorithm}')

    msen, pepn = evaluate_flow(flow_noc, flow)
    print(f'MSEN: {msen:.4f}, PEPN: {pepn:.4f}, runtime: {toc-tic:.3f}s')

    cv2.imshow(f'flow_{algorithm}', draw_flow(img_prev, flow))
    cv2.imshow(f'hsv_{algorithm}', draw_hsv(flow))
    cv2.waitKey(0)


def task2_1():
    # Video stabilization with Block Matching

    cap = cv2.VideoCapture('data/test1.mp4')
    out = "results/week4/test"

    if not os.path.exists(out):
        os.makedirs(out)
    block_matching_stabilization(cap, out, to_video=False, video_percentage=0.15)


def task2_2(method="point_feature"):
    # Off-the-shelf Stabilization

    cap = cv2.VideoCapture('data/shaky_videos/seattle.avi')
    out = f"results/week4/{method}/seattle"
    if not os.path.exists(out):
        os.makedirs(out)

    if method == "point_feature":
        smooth_radius = 5  # play a bit with this paramenter
        point_feature_matching(cap, smooth_radius, out, to_video=False, video_percentage=0.3)

    elif method == "mesh_flow":
        mesh_flow_main(cap, out, video_percentage=0.3)


def task3_1(video_percentage=1):
    # Tracking with optical flow

    cap = cv2.VideoCapture('data/AICity_data/train/S03/c010/vdo.avi')
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    save_path = 'results/week4/task_31'
    os.makedirs(save_path, exist_ok=True)

    reader = AICityChallengeAnnotationReader(path='data/ai_challenge_s03_c010-full_annotation.xml')
    gt = reader.get_annotations(classes=['car'])
    reader = AICityChallengeAnnotationReader(path='data/AICity_data/train/S03/c010/det/det_mask_rcnn.txt')
    dets = reader.get_annotations(classes=['car'])

    if save_path:
        writer = imageio.get_writer(os.path.join(save_path, f'task31.gif'), fps=fps)

    accumulator = MOTAcumulator()
    y_true = []
    y_pred = []
    y_pred_refined = []
    tracks = []
    max_track = 0
    previous_frame = None
    end = int(n_frames * video_percentage)
    for i, frame in tqdm(enumerate(dets.keys())):
        if i == end:
            break

        if save_path:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, img = cap.read()

        if i == 0:
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

        detections_on_frame = dets.get(frame, [])
        tracks, frame_tracks, max_track = update_tracks_by_overlap(tracks,
                                                                   detections_on_frame,
                                                                   max_track,
                                                                   refinement=False, 
                                                                   optical_flow=optical_flow)

        frame_detections = []
        for track in frame_tracks:
            det = track.last_detection()
            frame_detections.append(det)
            if save_path:
                cv2.rectangle(img, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ybr)), track.color, 2)
                cv2.rectangle(img, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ytl) - 15), track.color, -2)
                cv2.putText(img, str(det.id), (int(det.xtl), int(det.ytl)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                for dd in track.detections:
                    cv2.circle(img, dd.center, 5, track.color, -1)

        y_pred_refined.append(frame_detections)
        y_pred.append(detections_on_frame)
        y_true.append(gt.get(frame, []))

        accumulator.update(y_true[-1], y_pred_refined[-1])

        if save_path:
            writer.append_data(cv2.resize(img, (600, 350)))

    cv2.destroyAllWindows()
    if save_path:
        writer.close()

    ap, prec, rec = mean_average_precision(y_true, y_pred, classes=['car'], sort_method='score')
    print(f'Original AP: {ap:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}')
    ap, prec, rec = mean_average_precision(y_true, y_pred_refined, classes=['car'], sort_method='score')
    print(f'After refinement AP: {ap:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}')
    print('\nAdditional metrics:')
    print(accumulator.get_idf1())


if __name__ == '__main__':
    # task1_1()
    task1_2(algorithm='pyflow')
    # task2_2(method="mesh_flow")
