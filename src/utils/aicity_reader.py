import os
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
                parked = False
            annotations.append(Detection(
                int(box['@frame']),
                int(id),
                label,
                parked,
                float(box['@xtl']),
                float(box['@ytl']),
                float(box['@xbr']),
                float(box['@ybr'])
            ))

    return annotations


def parse_annotations_from_txt(path):
    """
    MOTChallenge format [frame,ID,left,top,width,height,conf,-1,-1,-1]
    """

    with open(path) as f:
        lines = f.readlines()

    annotations = []
    for line in lines:
        data = line.split(',')
        annotations.append(Detection(
            int(data[0]) - 1,
            int(data[1]),
            'car',
            None,
            float(data[2]),
            float(data[3]),
            float(data[2]) + float(data[4]),
            float(data[3]) + float(data[5]),
            float(data[6])
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


class AICityChallengeAnnotationReader:

    def __init__(self, path):
        self.annotations = parse_annotations(path)
        self.classes = np.unique([detection.label for detection in self.annotations])

    def get_annotations(self, classes=None, noise_params=None, group_by_frame=True, only_not_parked=False):
        """
        Returns:
            detections: {frame: [Detection,...]} if group_by_frame=True
        """

        if classes is None:
            classes = self.classes

        detections = []
        for detection in self.annotations:
            if detection.label in classes:  # filter by class
                if only_not_parked and detection.parked:
                    continue
                if noise_params:  # add noise
                    if np.random.random() > noise_params['drop']:
                        detection.xtl += np.random.normal(noise_params['mean'], noise_params['std'], 1)[0]
                        detection.ytl += np.random.normal(noise_params['mean'], noise_params['std'], 1)[0]
                        detection.xbr += np.random.normal(noise_params['mean'], noise_params['std'], 1)[0]
                        detection.ybr += np.random.normal(noise_params['mean'], noise_params['std'], 1)[0]
                        detections.append(detection)
                else:
                    detections.append(detection)

        if group_by_frame:
            grouped = defaultdict(list)
            for detection in detections:
                grouped[detection.frame].append(detection)
            detections = OrderedDict(sorted(grouped.items()))

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
    for box in gt[frame]:
        cv2.rectangle(img, (int(box[1]), int(box[2])), (int(box[3]), int(box[4])), (0, 255, 0), 2)
    for box in det[frame]:
        cv2.rectangle(img, (int(box[1]), int(box[2])), (int(box[3]), int(box[4])), (0, 0, 255), 2)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
