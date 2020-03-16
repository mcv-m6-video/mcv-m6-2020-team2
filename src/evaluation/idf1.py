import motmetrics as mm
import numpy as np


class MOTAcumulator:

    def __init__(self):
        self.acc = mm.MOTAccumulator(auto_id=True)

    def update(self, true, pred):
        true_ids = [i.id for i in true]
        pred_ids = [i.id for i in pred]

        true_c = np.array([(int((i.xtl+i.xbr)/2), int((i.ytl+i.ybr)/2)) for i in true])
        pred_c = np.array([(int((i.xtl+i.xbr)/2), int((i.ytl+i.ybr)/2)) for i in pred])

        if true_ids and pred_ids:
            distances = [[np.sqrt(np.sum((i-j)**2)) for j in pred_c] for i in true_c]
            self.acc.update(
                true_ids,
                pred_ids,
                distances
                )
        else:
            self.acc.update(
                true_ids,
                pred_ids,
                []
                )

    def compute(self):
        mh = mm.metrics.create()
        summary = mh.compute(self.acc, metrics=['idf1'], name='acc')
        return summary
