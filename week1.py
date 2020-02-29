import numpy as np

from src.evaluation.mean_average_precision import get_mAP
from src.utils.readers import AICityChallengeGTAnnotationReader


def task1():
    # Task 1.1
    noisy_params = {
        'prob_delete': 0.05,
        'mean': 0,
        'std': 10  # Video is 1920x1080
    }

    gt_reader = AICityChallengeGTAnnotationReader()
    gt = gt_reader.get_gt(classes=['car'], group_by_frame=True, only_bounding_boxes=True)
    gt_noisy = gt_reader.get_gt(classes=['car'], noisy=noisy_params, group_by_frame=True, only_bounding_boxes=True)

    mAP_all_frames = get_mAP(gt_noisy, gt, confidence_scores=False)
    mAP_avg_across_frames = np.mean([v for k, v in mAP_all_frames.items()])

    print('mAP averaged across all frames: {:.4f}'.format(mAP_avg_across_frames))

    # TODO: TASK 1.2


if __name__ == '__main__':
    task1()
