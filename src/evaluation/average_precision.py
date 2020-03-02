import numpy as np

from .intersection_over_union import vec_intersecion_over_union


def mean_average_precision(y_true, y_pred, classes=None):
    """
    Mean Average Precision across classes.

    Args:
        y_true: [[[cls,xtl,ytl,xbr,ybr],...],...]
        y_pred: [[[cls,xtl,ytl,xbr,ybr,score],...],...]
        classes: list of considered classes.
    """

    if classes is None:
        classes = np.unique([box[0] for boxlist in y_true for box in boxlist])

    aps = []
    for cls in classes:
        # filter by class
        y_true_cls = [[box[1:] for box in boxlist if box[0] == cls] for boxlist in y_true]
        y_pred_cls = [[box[1:] for box in boxlist if box[0] == cls] for boxlist in y_pred]
        ap = average_precision(y_true_cls, y_pred_cls)
        aps.append(ap)
    map = np.mean(aps) if aps else 0

    return map


def average_precision(y_true, y_pred):
    """
    Average Precision with or without confidence scores.

    Args:
        y_true: [[[xtl,ytl,xbr,ybr],...],...]
        y_pred: [[[xtl,ytl,xbr,ybr(,score)],...],...]
    """

    y_pred = [[i] + box for i in range(len(y_pred)) for box in y_pred[i]]  # flatten
    if len(y_pred[0]) > 5:
        # sort by confidence
        sorted_ind = np.argsort([-box[-1] for box in y_pred])
        y_pred_sorted = [y_pred[i][:-1] for i in sorted_ind]
        ap = voc_ap(y_true, y_pred_sorted)
    else:
        # average metrics across n random orderings
        n = 10
        aps = []
        for _ in range(n):
            shuffled_ind = np.random.permutation(len(y_pred))
            y_pred_shuffled = [y_pred[i] for i in shuffled_ind]
            ap = voc_ap(y_true, y_pred_shuffled)
            aps.append(ap)
        ap = np.mean(aps)
    return ap


# Below code is modified from
# https://github.com/facebookresearch/detectron2/blob/master/detectron2/evaluation/pascal_voc_evaluation.py

def voc_ap(y_true, y_pred, ovthresh=0.5):
    """
    Average Precision as defined by PASCAL VOC (11-point method).

    Args:
        y_true: [[[xtl,ytl,xbr,ybr],...],...]
        y_pred: [[id,xtl,ytl,xbr,ybr],...]
        ovthresh: overlap threshold.
    """

    class_recs = []
    npos = 0
    for R in y_true:
        bbox = np.array(R)
        det = [False] * len(R)
        npos += len(R)
        class_recs.append({"bbox": bbox, "det": det})

    image_ids = [box[0] for box in y_pred]
    BB = np.array([box[1:] for box in y_pred]).reshape(-1, 4)

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            overlaps = vec_intersecion_over_union(BBGT, bb[None, :])
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R["det"][jmax]:
                tp[d] = 1.0
                R["det"][jmax] = 1
            else:
                fp[d] = 1.0
        else:
            fp[d] = 1.0

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    # compute VOC AP using 11 point metric
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(rec >= t) == 0:
            p = 0
        else:
            p = np.max(prec[rec >= t])
        ap = ap + p / 11.0

    return ap
