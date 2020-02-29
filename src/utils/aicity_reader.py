from itertools import chain
from collections import defaultdict, OrderedDict

import numpy as np
import xmltodict


def parse_annotations(path):
    with open(path) as f:
        tracks = xmltodict.parse(f.read())['annotations']['track']

    annotations = []
    for track in tracks:
        id = track['@id']
        label = track['@label']
        boxes = track['box']
        for box in boxes:
            annotations.append([
                int(id),
                label,
                int(box['@frame']),
                float(box['@xtl']),
                float(box['@ytl']),
                float(box['@xbr']),
                float(box['@ybr']),
                int(box['@outside']),
                int(box['@occluded']),
                int(box['@keyframe'])
            ])

    return annotations


class AICityChallengeAnnotationReader:

    def __init__(self, path):
        self.gts = parse_annotations(path)
        self.classes = np.unique([x[1] for x in self.gts])

    def get_gt(self, classes=None, noise_params=None, group_by_frame=False, boxes_only=False):
        if classes is None:
            classes = self.classes

        res = []
        for gt in self.gts:
            if gt[1] in classes:  # filter by class
                if noise_params:  # add noise
                    if np.random.random() > noise_params['drop']:
                        box_noisy = list(gt[3:7] + np.random.normal(noise_params['mean'], noise_params['std'], 4))
                        res.append(list(chain.from_iterable([gt[:3], box_noisy, gt[7:]])))
                else:
                    res.append(gt[:])

        if group_by_frame:
            grouped = defaultdict(list)
            for gt in res:
                frame = gt[2]
                if boxes_only:
                    gt = gt[3:7]
                else:
                    del gt[2]
                grouped[frame].append(gt)
            res = OrderedDict(sorted(grouped.items()))

        return res


if __name__ == '__main__':
    reader = AICityChallengeAnnotationReader(path='../../data/ai_challenge_s03_c010-full_annotation.xml')
    gt = reader.get_gt(classes=['car'], group_by_frame=True, boxes_only=True)
    gt_noisy = reader.get_gt(classes=['car'], noise_params={'drop': 0.05, 'mean': 0, 'std': 10}, group_by_frame=True, boxes_only=True)

    import cv2
    cap = cv2.VideoCapture('../../data/AICity_data/train/S03/c010/vdo.avi')
    frame = np.random.randint(0, len(gt))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, img = cap.read()
    for box in gt[frame]:
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
    for box in gt_noisy[frame]:
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
