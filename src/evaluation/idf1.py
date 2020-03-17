import numpy as np
import motmetrics as mm
from sklearn.metrics.pairwise import pairwise_distances


class MOTAcumulator:

    def __init__(self):
        self.acc = mm.MOTAccumulator(auto_id=True)

    def update(self, y_true, y_pred):
        X = np.array([[(d.xtl+d.xbr)/2, (d.ytl+d.ybr)/2] for d in y_true])
        Y = np.array([[(d.xtl+d.xbr)/2, (d.ytl+d.ybr)/2] for d in y_pred])
        dists = pairwise_distances(X, Y, metric='euclidean')
        self.acc.update([i.id for i in y_true], [i.id for i in y_pred], dists)

    def compute(self):
        mh = mm.metrics.create()
        return mh.compute(self.acc, metrics=['idf1'], name='acc')
