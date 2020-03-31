import os
from copy import deepcopy
from collections import defaultdict, OrderedDict
import numpy as np
import xmltodict

from src.utils.detection import Detection


def parse_annotations_from_xml(path):
    with open(path) as f:
        tracks = xmltodict.parse(f.read())['annotations']['track']

    annotations = []
    for track in tracks:
        id = track['@id']
        label = track['@label']
        boxes = track['box']
        for box in boxes:
            if label == 'car':
                parked = box['attribute']['#text'].lower() == 'true'
            else:
                parked = None
            annotations.append(Detection(
                frame=int(box['@frame']),
                id=int(id),
                label=label,
                xtl=float(box['@xtl']),
                ytl=float(box['@ytl']),
                xbr=float(box['@xbr']),
                ybr=float(box['@ybr']),
                parked=parked
            ))

    return annotations

def parse_annotations_from_txt(path):
    """
    MOTChallenge format [frame, ID, left, top, width, height, conf, -1, -1, -1]
    """

    with open(path) as f:
        lines = f.readlines()

    annotations = []
    for line in lines:
        data = line.split(',')
        annotations.append(Detection(
            frame=int(data[0])-1,
            id=int(data[1]),
            label='car',
            xtl=float(data[2]),
            ytl=float(data[3]),
            xbr=float(data[2])+float(data[4]),
            ybr=float(data[3])+float(data[5]),
            score=float(data[6])
        ))

    return annotations

def parse_annotations(path):
    root, ext = os.path.splitext(path)
    if ext == ".xml":
        return parse_annotations_from_xml(path)
    elif ext == ".txt":
        return parse_annotations_from_txt(path)
    else:
        raise ValueError(f'Invalid file extension: {ext}')

def group_detections_by_frame(detections):
    grouped = defaultdict(list)
    for det in detections:
        grouped[det.frame].append(det)
    return OrderedDict(sorted(grouped.items()))


class AICityChallengeAnnotationReader:

    def __init__(self, path):
        self.annotations = parse_annotations(path)
        self.classes = np.unique([det.label for det in self.annotations])

    def get_annotations(self, classes=None, noise_params=None, group_by_frame=True, only_not_parked=False):
        """
        Returns:
            detections: {frame: [Detection,...]} if group_by_frame=True
        """

        if classes is None:
            classes = self.classes

        detections = []
        for det in self.annotations:
            if det.label in classes:  # filter by class
                if only_not_parked and det.parked:
                    continue
                d = deepcopy(det)
                if noise_params:  # add noise
                    if np.random.random() > noise_params['drop']:
                        box_noisy = d.bbox + np.random.normal(noise_params['mean'], noise_params['std'], 4)
                        d.xtl = box_noisy[0]
                        d.ytl = box_noisy[1]
                        d.xbr = box_noisy[2]
                        d.ybr = box_noisy[3]
                        detections.append(d)
                else:
                    detections.append(d)

        if group_by_frame:
            detections = group_detections_by_frame(detections)

        return detections


if __name__ == '__main__':
    reader = AICityChallengeAnnotationReader(path='../../data/ai_challenge_s03_c010-full_annotation.xml')
    gt = reader.get_annotations(classes=['car'])
    #gt_noisy = reader.get_annotations(classes=['car'], noise_params={'drop': 0.05, 'mean': 0, 'std': 10})
    reader = AICityChallengeAnnotationReader(path='../../data/AICity_data/train/S03/c010/det/det_mask_rcnn.txt')
    det = reader.get_annotations(classes=['car'])

    import cv2
    cap = cv2.VideoCapture('../../data/AICity_data/train/S03/c010/vdo.avi')
    frame = np.random.randint(0, len(gt))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, img = cap.read()
    for d in gt[frame]:
        cv2.rectangle(img, (int(d.xtl), int(d.ytl)), (int(d.xbr), int(d.ybr)), (0, 255, 0), 2)
    for d in det[frame]:
        cv2.rectangle(img, (int(d.xtl), int(d.ytl)), (int(d.xbr), int(d.ybr)), (0, 0, 255), 2)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
