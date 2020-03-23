import os
import time
from itertools import product

import cv2
import pandas as pd

from src.optical_flow.block_matching_flow import read_flow, block_matching_flow, evaluate_flow
from src.video_stabilization.block_matching_stabilization import block_matching_stabilization
from src.video_stabilization.point_feature_matching import point_feature_matching


def task1_1():
    # Optical Flow with Block Matching

    img_prev = cv2.imread('data/data_stereo_flow/training/image_0/000045_10.png', cv2.IMREAD_GRAYSCALE)
    img_next = cv2.imread('data/data_stereo_flow/training/image_0/000045_11.png', cv2.IMREAD_GRAYSCALE)
    flow_noc = read_flow('data/data_stereo_flow/training/flow_noc/000045_10.png')

    motion_type = ['forward', 'backward']
    search_area = [16, 32, 64, 128]
    block_size = [4, 8, 16]

    data = []
    for m, p, n in product(motion_type, search_area, block_size):
        tic = time.time()
        flow = block_matching_flow(img_prev, img_next, motion_type=m, search_area=p, block_size=n, algorithm='corr')
        toc = time.time()
        msen, pepn = evaluate_flow(flow_noc, flow)
        data.append([m, p, n, msen, pepn, toc-tic])
    df = pd.DataFrame(data, columns=['motion_type', 'search_area', 'block_size', 'msen', 'pepn', 'runtime'])
    print(df)


def task1_2():
    # Off-the-shelf Optical Flow
    pass


def task2_1():
    # Video stabilization with Block Matching

    cap = cv2.VideoCapture('data/test1.mp4')
    out = "results/week4/test"

    if not os.path.exists(out):
        os.makedirs(out)
    block_matching_stabilization(cap, out, to_video=False, video_percentage=0.15)


def task2_2():
    # Off-the-shelf Stabilization

    cap = cv2.VideoCapture('data/pati.mp4')
    out = "results/week4/patio"

    if not os.path.exists(out):
        os.makedirs(out)
    point_feature_matching(cap, out, to_video=False, video_percentage=0.3)


def task3_1():
    # Object Tracking with Optical Flow
    pass


if __name__ == '__main__':
    task1_1()
    # task2_2()
