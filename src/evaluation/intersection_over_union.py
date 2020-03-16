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

    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)

    # intersection
    ixmin = np.maximum(x11, np.transpose(x21))
    iymin = np.maximum(y11, np.transpose(y21))
    ixmax = np.minimum(x12, np.transpose(x22))
    iymax = np.minimum(y12, np.transpose(y22))
    iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
    ih = np.maximum(iymax - iymin + 1.0, 0.0)
    inters = iw * ih

    # union
    area1 = (x12 - x11 + 1.0) * (y12 - y11 + 1.0)
    area2 = (x22 - x21 + 1.0) * (y22 - y21 + 1.0)
    uni = area1 + np.transpose(area2) - inters

    overlaps = inters / uni

    return overlaps


def mean_intersection_over_union(boxes1, boxes2):
    boxes1 = np.array(boxes1).reshape(-1, 4)
    boxes2 = np.array(boxes2).reshape(-1, 4)
    overlaps = vec_intersecion_over_union(boxes1, boxes2)
    # for each gt (rows) select the max overlap with any detection (columns)
    return np.mean(np.max(overlaps, axis=1))


def find_closest_bb(bb, bb_list):
    """
    Returns the bounding box in bb_list closest to bb
    """

    max_iou = 0
    best_bb = bb_list[0]
    for b in bb_list:
        iou = bb_intersecion_over_union(bb, [b.xtl, b.ytl, b.xbr, b.ybr])
        # print('\t',bb,[b.xtl, b.ytl, b.xbr, b.ybr],iou)
        if iou > max_iou:
            max_iou = iou
            best_bb = b
    return best_bb
