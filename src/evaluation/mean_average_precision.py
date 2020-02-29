from random import shuffle
import numpy as np

from .intersection_over_union import IoU_from_bbs


def get_mAP(x, gt, confidence_scores=True):
    if confidence_scores:
        mAP = _mAP_with_confidence(x, gt)
    else:
        mAP = _mAP_without_confidence(x, gt)
    return mAP

def _mAP_with_confidence(x, gt):
    # TODO: Sort x by confidence and compute mAP
    mAPS = {}
    return mAPS

def _mAP_without_confidence(x, gt):
    N = 10 # Number of random ranks

    APs = {}
    for i in range(N):
        x2 = {}
        for k, v in x.items():
            shuffle(v)
            x2[k] = v
        AP = _compute_AP(x2, gt)
        # Append AP of each frame to a list
        for k, v in AP.items():
            if k not in APs.keys():
                APs[k] = []
            APs[k].append(v)
    # Compute mean AP for each frame
    mAPs = {}
    for k, v in APs.items():
        mAPs[k] = np.mean(v)

    return mAPs

def _compute_AP(x, gt):
    """
        x: {int(frame_num): [[x0,y0,x1,y1],[...]]}
        gt: {int(frame_num): [[x0,y0,x1,y1],[...]]}
    """
    iou_thresh = 0.5

    # For each frame
    APs = {}
    for frame_num, bb_list in x.items():
        bb_list_gt = gt[frame_num]
        n_items = len(bb_list_gt)

        # Compute correctness of bounding boxes
        correctness = [max([IoU_from_bbs(i, j) for j in bb_list_gt]) > iou_thresh for i in bb_list]

        # Compute precision and recall at each step
        precision = []
        recall = []
        TP = 0
        for i, c in enumerate(correctness):
            if c:
                TP += 1
            precision.append(TP/(i+1))
            recall.append(TP/n_items)
        precision.append(0)
        recall.append(1)

        # Remove unnecessary points
        to_remove = []
        max_precision = -1
        for i, value in enumerate(precision[::-1]):
            if value < max_precision:
                to_remove.append(len(precision)-1-i) # Append real index, not reverse index
            else:
                max_precision = value
        precision = [value for i, value in enumerate(precision) if i not in to_remove]
        recall = [value for i, value in enumerate(recall) if i not in to_remove]

        # Compute Pascal VOC AP
        ps = []
        for x_val in np.linspace(0, 1, 11):
            recall_val = min([i for i in recall if i-x_val >= 0])
            ps.append(precision[recall.index(recall_val)])

        APs[frame_num] = np.mean(ps)

    return APs
