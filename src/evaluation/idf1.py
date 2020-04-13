import motmetrics as mm
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

from utils.aicity_reader import parse_annotations_from_txt, group_by_frame


class MOTAcumulator:

    def __init__(self):
        self.acc = mm.MOTAccumulator(auto_id=True)

    def update(self, y_true, y_pred):
        X = np.array([det.center for det in y_true])
        Y = np.array([det.center for det in y_pred])

        if len(X) > 0 and len(Y) > 0:
            dists = pairwise_distances(X, Y, metric='euclidean')
        else:
            dists = np.array([])

        self.acc.update(
            [det.id for det in y_true],
            [det.id for det in y_pred],
            dists
        )

    def get_idf1(self):
        mh = mm.metrics.create()
        summary = mh.compute(self.acc, metrics=['idf1'], name='acc')
        return summary['idf1']['acc']

    def get_metrics(self):
        mh = mm.metrics.create()
        summary = mh.compute(self.acc, metrics=['idf1', 'idp', 'precision', 'recall'], name='acc')
        return summary


def compute_idf1(true_path, pred_path):
    dets_true = group_by_frame(parse_annotations_from_txt(true_path))
    dets_pred = group_by_frame(parse_annotations_from_txt(pred_path))

    accumulator = MOTAcumulator()
    for frame in dets_true.keys():
        y_true = dets_true.get(frame, [])
        y_pred = dets_pred.get(frame, [])
        accumulator.update(y_true, y_pred)

    return accumulator.get_idf1()
