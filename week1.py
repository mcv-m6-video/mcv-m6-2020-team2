import os

import matplotlib.pyplot as plt

from src.utils.aicity_reader import AICityChallengeAnnotationReader
from src.evaluation.average_precision import mean_average_precision
from src.evaluation.intersection_over_union import mean_intersection_over_union

from src.evaluation.optical_flow_evaluation import get_msen_pepn
from src.utils.io_optical_flow import read_flow_field, read_grayscale_image


def task1():
    # Task 1.1
    reader = AICityChallengeAnnotationReader(path='data/ai_challenge_s03_c010-full_annotation.xml')
    gt = reader.get_gt(classes=['car'], group_by_frame=True, boxes_only=True)

    noise_params = {
        'drop': 0.05,
        'mean': 0,
        'std': 10  # video is 1920x1080
    }
    gt_noisy = reader.get_gt(classes=['car'], noise_params=noise_params, group_by_frame=True, boxes_only=True)

    frames = sorted(list(set(gt) & set(gt_noisy)))
    y_true = []
    y_pred = []
    for frame in frames:
        y_true.append(gt[frame])
        y_pred.append(gt_noisy[frame])

    map = mean_average_precision(y_true, y_pred)
    print(f'mAP: {map:.4f}')

    # TODO: Task 1.2


def task2():
    reader = AICityChallengeAnnotationReader(path='data/ai_challenge_s03_c010-full_annotation.xml')
    gt = reader.get_gt(classes=['car'], group_by_frame=True, boxes_only=True)
    noise_params = {'drop': 0.05, 'mean': 0, 'std': 10}
    gt_noisy = reader.get_gt(classes=['car'], noise_params=noise_params, group_by_frame=True, boxes_only=True)

    frames = sorted(list(set(gt) & set(gt_noisy)))
    overlaps = []
    for frame in frames:
        boxes1 = [box[1:] for box in gt[frame]]
        boxes2 = [box[1:] for box in gt_noisy[frame]]
        iou = mean_intersection_over_union(boxes1, boxes2)
        overlaps.append(iou)

    plt.plot(frames, overlaps)
    plt.ylim(0, 1)
    plt.xlabel('#frame')
    plt.ylabel('mean IoU')
    plt.title('noisy annotation')
    plt.show()


def task3():
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

        msen, pepn = get_msen_pepn(flow_pred, flow_gt, frame_id=im_idx, th=3, plot=True)
        print(f'SEQ-{im_idx}\n  MSEN: {round(msen, 2)}\n  PEPN: {round(pepn, 2)}%')


if __name__ == '__main__':
    #task1()
    task2()
    #task3()
