import os
import cv2

from video_stabilization.point_feature_matching import point_feature_matching


def task1_1():
    # Optical Flow Estimation
    return None

def task1_2():
    # Off the Shelf optical Flow
    return None

def task2_1():
    # Video Stabilization
    return None

def task2_2():
    # Off the shelf video stabilization

    cap = cv2.VideoCapture('data/pati.mp4')
    out =  "results/week4/patio"

    if not os.path.exists(out):
        os.makedirs(out)
    point_feature_matching(cap, out, to_video=False, video_percentage=0.3)


def task3_1():
    # Tracking with optical flow
    return None


if __name__ == '__main__':
    task2_2()