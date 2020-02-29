import os

import numpy as np

from src.utils.aicity_reader import AICityChallengeAnnotationReader
from src.evaluation.mean_average_precision import get_mAP

from src.evaluation.optical_flow_evaluation import get_msen_pepn
from src.utils.io_optical_flow import read_flow_field, read_grayscale_image


def task1():
    # Task 1.1
    noise_params = {
        'drop': 0.05,
        'mean': 0,
        'std': 10  # video is 1920x1080
    }

    reader = AICityChallengeAnnotationReader(path='data/ai_challenge_s03_c010-full_annotation.xml')
    gt = reader.get_gt(classes=['car'], group_by_frame=True, boxes_only=True)
    gt_noisy = reader.get_gt(classes=['car'], noise_params=noise_params, group_by_frame=True, boxes_only=True)

    mAP_all_frames = get_mAP(gt_noisy, gt, confidence_scores=False)
    mAP_avg_across_frames = np.mean([v for k, v in mAP_all_frames.items()])

    print('mAP averaged across all frames: {:.4f}'.format(mAP_avg_across_frames))

    # TODO: TASK 1.2


def task2():
    pred_path = 'data/optical_flow_results/'
    kitti_data = 'data/data_stereo_flow/training/'
    images = ['045', '157']
    plot = False

    for im_idx in images:
        im_name = f'000{im_idx}_10.png'

        flow_estimation = os.path.join(pred_path, f'LKflow_{im_name}')
        im_path = os.path.join(kitti_data, f'image_0/{im_name}')
        gt_non_occ = os.path.join(kitti_data,f'flow_noc/{im_name}')

        image_gray = read_grayscale_image(im_path, frame_id=im_idx, plot=plot)
        flow_pred = read_flow_field(flow_estimation, frame_id=im_idx, plot=plot)
        flow_gt = read_flow_field(gt_non_occ, frame_id=im_idx, plot=plot)

        msen, pepn = get_msen_pepn(flow_pred, flow_gt, frame_id=im_idx, th=3, plot=True)
        print(f'SEQ-{im_idx}\n  MSEN: {round(msen, 2)}\n  PEPN: {round(pepn, 2)}%')


if __name__ == '__main__':
    task1()
    task2()
