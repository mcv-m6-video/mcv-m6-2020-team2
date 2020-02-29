import numpy as np

from src.utils.aicity_reader import AICityChallengeAnnotationReader
from src.evaluation.mean_average_precision import get_mAP


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


if __name__ == '__main__':
    task1()
