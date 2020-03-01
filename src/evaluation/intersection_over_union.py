import numpy as np


def bb_intersecion_over_union(bb1, bb2):
    """
    Compute overlap between two boxes.

    Args:
        bb1: [xtl,ytl,xbr,ybr]
        bb2: [xtl,ytl,xbr,ybr]
    """

    # determine intersection rectangle coordinates
    left = max(bb1[0], bb2[0])
    top = max(bb1[1], bb2[1])
    right = min(bb1[2], bb2[2])
    bottom = min(bb1[3], bb2[3])

    # check non-overlapping rectangle
    if right < left or bottom < top:
        return 0

    # compute intersection area
    intersection_area = (right - left) * (bottom - top)

    # compute area of bbs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute IoU
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    return iou


def vec_intersecion_over_union(boxes1, boxes2):
    """
    Compute overlaps between two sets of boxes.

    Args:
        boxes1: [[xtl,ytl,xbr,ybr],...]
        boxes2: [[xtl,ytl,xbr,ybr],...]
    Returns:
        overlaps: matrix of pairwise overlaps.
    """

    # intersection
    ixmin = np.maximum(boxes1[:, 0], np.transpose(boxes2[:, 0]))
    iymin = np.maximum(boxes1[:, 1], np.transpose(boxes2[:, 1]))
    ixmax = np.minimum(boxes1[:, 2], np.transpose(boxes2[:, 2]))
    iymax = np.minimum(boxes1[:, 3], np.transpose(boxes2[:, 3]))
    iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
    ih = np.maximum(iymax - iymin + 1.0, 0.0)
    inters = iw * ih

    # union
    area1 = (boxes1[:, 2] - boxes1[:, 0] + 1.0) * (boxes1[:, 3] - boxes1[:, 1] + 1.0)
    area2 = (boxes2[:, 2] - boxes2[:, 0] + 1.0) * (boxes2[:, 3] - boxes2[:, 1] + 1.0)
    uni = area1 + area2 - inters

    overlaps = inters / uni

    return overlaps
