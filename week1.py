import os

import numpy as np
import matplotlib.pyplot as plt

from src.utils.aicity_reader import AICityChallengeAnnotationReader
from src.evaluation.average_precision import mean_average_precision
from src.utils.plotutils import video_iou_plot

from src.evaluation.optical_flow_evaluation import get_msen_pepn
from src.utils.io_optical_flow import read_flow_field, read_grayscale_image
from src.utils.optical_flow_visualization import plot_optical_flow


def task1_1(save_path=None):
    reader = AICityChallengeAnnotationReader(path='data/ai_challenge_s03_c010-full_annotation.xml')
    gt = reader.get_annotations(classes=['car'])

    # add probability to delete bounding boxes
    drop_values = np.linspace(0, 1, 11)
    maps = []
    for drop in drop_values:
        noise_params = {'drop': drop, 'mean': 0, 'std': 0}
        gt_noisy = reader.get_annotations(classes=['car'], noise_params=noise_params)

        frames = sorted(list(set(gt) & set(gt_noisy)))
        y_true = []
        y_pred = []
        for frame in frames:
            y_true.append(gt[frame])
            y_pred.append(gt_noisy[frame])

        map = mean_average_precision(y_true, y_pred)
        maps.append(map)

    plt.plot(drop_values, maps)
    plt.xticks(drop_values)
    plt.xlabel('drop prob')
    plt.ylabel('mAP')
    plt.show()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'map_drop_bbox.png'))

    # add noise to the size and position of bounding boxes
    std_values = np.linspace(0, 100, 11)
    maps = []
    for std in std_values:
        noise_params = {'drop': 0, 'mean': 0, 'std': std}
        gt_noisy = reader.get_annotations(classes=['car'], noise_params=noise_params)

        frames = sorted(list(set(gt) & set(gt_noisy)))
        y_true = []
        y_pred = []
        for frame in frames:
            y_true.append(gt[frame])
            y_pred.append(gt_noisy[frame])

        map = mean_average_precision(y_true, y_pred)
        maps.append(map)

    plt.xlabel('std')
    plt.ylabel('mAP')
    plt.xticks(std_values)
    plt.plot(std_values, maps)
    plt.show()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'map_noisy_bbox.png'))


def task1_2():
    reader = AICityChallengeAnnotationReader(path='data/ai_challenge_s03_c010-full_annotation.xml')
    gt = reader.get_annotations(classes=['car'])

    for detector in ['mask_rcnn', 'ssd512', 'yolo3']:
        reader = AICityChallengeAnnotationReader(path=f'data/AICity_data/train/S03/c010/det/det_{detector}.txt')
        det = reader.get_annotations(classes=['car'])

        frames = sorted(list(set(gt) & set(det)))
        y_true = []
        y_pred = []
        for frame in frames:
            y_true.append(gt[frame])
            y_pred.append(det[frame])

        map = mean_average_precision(y_true, y_pred)
        print(f'{detector} mAP: {map:.4f}')


def task2(save_path=None):
    reader = AICityChallengeAnnotationReader(path='data/ai_challenge_s03_c010-full_annotation.xml')
    gt = reader.get_annotations(classes=['car'])

    noise_params = {'drop': 0.05, 'mean': 0, 'std': 10}
    gt_noisy = reader.get_annotations(classes=['car'], noise_params=noise_params)
    video_iou_plot(gt, gt_noisy, video_path='data/AICity_data/train/S03/c010/vdo.avi', title='noisy annotations',
                   save_path=save_path)

    for detector in ['mask_rcnn', 'ssd512', 'yolo3']:
        reader = AICityChallengeAnnotationReader(path=f'data/AICity_data/train/S03/c010/det/det_{detector}.txt')
        det = reader.get_annotations(classes=['car'])
        video_iou_plot(gt, det, video_path='data/AICity_data/train/S03/c010/vdo.avi', title=f'{detector} detections',
                       save_path=save_path)


def task3_4(path_results=None):
    pred_path = 'data/optical_flow_results/'
    kitti_data = 'data/data_stereo_flow/training/'
    images = ['045', '157']
    plot = False

    for im_idx in images:
        im_name = f'000{im_idx}_10.png'

        flow_estimation = os.path.join(pred_path, f'LKflow_{im_name}')
        im_path = os.path.join(kitti_data, f'image_0/{im_name}')
        gt_non_occ = os.path.join(kitti_data, f'flow_noc/{im_name}')

        image_gray = read_grayscale_image(im_path, frame_id=im_idx, plot=plot)
        flow_pred = read_flow_field(flow_estimation, frame_id=im_idx, plot=plot)
        flow_gt = read_flow_field(gt_non_occ, frame_id=im_idx, plot=plot)

        print('Computing MSEN and PEPN')
        msen, pepn = get_msen_pepn(flow_pred, flow_gt, frame_id=im_idx, th=3, plot=plot)
        print(f'SEQ-{im_idx}\n  MSEN: {round(msen, 4)}\n  PEPN: {round(pepn, 4)}%')

        plot_optical_flow(image_gray, flow_gt[:, :, 0:2], 'GT', im_idx, 10, path=path_results)
        plot_optical_flow(image_gray, flow_pred[:, :, 0:2], 'PRED', im_idx, 10, path=path_results)


if __name__ == '__main__':
    # task1_1()
    # task1_2()
    task2()
    # task3_4()
