import os
from glob import glob

import numpy as np
import motmetrics as mm
from sklearn.metrics.pairwise import pairwise_distances

from src.utils.aicity_reader import AICityChallengeAnnotationReader


class MOTAcumulator:

    def __init__(self):
        self.acc = mm.MOTAccumulator(auto_id=True)

    def update(self, y_true, y_pred):
        X = np.array([[(d.xtl+d.xbr)/2, (d.ytl+d.ybr)/2] for d in y_true])
        Y = np.array([[(d.xtl+d.xbr)/2, (d.ytl+d.ybr)/2] for d in y_pred])
        if len(X) == 0 or len(Y) == 0:
            dists = []
        else:
            dists = pairwise_distances(X, Y, metric='euclidean')
        self.acc.update([i.id for i in y_true], [i.id for i in y_pred], dists)

    def get_idf1(self):
        mh = mm.metrics.create()
        summary = mh.compute(self.acc, metrics=['idf1', 'idp', 'idr', 'recall', 'mostly_tracked', 'num_false_positives', 'num_frames', 'mota', 'motp'], name='acc')
        return summary


def get_idf1_from_dir(dir, sequence, method, gt_dir='data/AICity_data/train'):
    """
    Assumes the structure inside 'dir' to be:
        dir
        |____sequence1
        |    |____camera1
        |    |    |____method1.txt
        |    |    |____method2.txt
        |    |    |____...
        |    |____camera2
        |         |____method1.txt
        |         |____method2.txt
        |         |____...
        |____...

    Names of sequence and cameras must be the same as in the 'gt_dir' folder.
    """
    accumulator = MOTAcumulator()
    for cam_dets_file, gt_file in zip(glob(os.path.join(dir, sequence, '*', method + '.txt')),\
                                      glob(os.path.join(gt_dir, sequence, '*', 'gt', 'gt.txt'))):
        # Read files
        reader = AICityChallengeAnnotationReader(path=gt_file)
        gt = reader.get_annotations(classes=['car'])
        reader = AICityChallengeAnnotationReader(path=cam_dets_file)
        dets = reader.get_annotations(classes=['car'])

        # Iterate over detections and accumulate
        start = min(list(dets.keys()) + list(gt.keys()))
        end = max(list(dets.keys()) + list(gt.keys()))
        for frame in range(start, end):
            y_true = gt.get(frame, [])
            y_pred = dets.get(frame, [])
            accumulator.update(y_true, y_pred)

    return accumulator.get_idf1()
