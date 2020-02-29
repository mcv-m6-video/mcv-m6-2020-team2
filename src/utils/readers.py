import itertools
from copy import deepcopy
import xmltodict
import numpy as np


class AICityChallengeGTAnnotationReader():

    def __init__(self, path='./data/ai_challenge_s03_c010-full_annotation.xml'):
        self.path = path
        self.all_classes = set(['car', 'bike'])
        self.gt = []
        self.gt_var = None
        self.parse_annotations()

    def parse_annotations(self):
        """ Parse AICity_chalenge_s03_c010 annotation file. """

        with open(self.path) as f:
            tracks = xmltodict.parse(f.read())['annotations']['track']

        for track in tracks:
            id = track['@id']
            label = track['@label']
            bbs = track['box']
            for bb in bbs:
                self.gt.append([
                    int(id),
                    label,
                    int(bb['@frame']),
                    float(bb['@xtl']),
                    float(bb['@ytl']),
                    float(bb['@xbr']),
                    float(bb['@ybr']),
                    int(bb['@outside']),
                    int(bb['@occluded']),
                    int(bb['@keyframe'])
                ])

    def get_gt(self, classes=['car','bike'], noisy=None, group_by_frame=False, only_bounding_boxes=False):
        self.gt_var = deepcopy(self.gt)
        # Remove unwanted classes
        diff_classes = self.all_classes.difference(set(classes))
        if diff_classes:
            self.remove_classes(diff_classes)
        # Add noise
        if noisy is not None:
            self.make_noisy(noisy)
        # Group by frame
        if group_by_frame:
            self.group_by_frame(only_bounding_boxes)

        return self.gt_var

    def remove_classes(self, classes):
        self.gt_var = [i for i in self.gt_var if i[1] not in classes]

    def make_noisy(self, params):
        # Delete bounding boxes with prob_delete probability 
        self.gt_var = [i for i in self.gt_var if np.random.random() > params['prob_delete']]

        # Make remaining bounding boxes noisy
        self.gt_var = [list(itertools.chain.from_iterable([i[:3], list(i[3:7]+np.random.normal(params['mean'], params['std'], 4)), i[7:]])) for i in self.gt_var]

    def group_by_frame(self, only_bounding_boxes):
        gt_var = {}
        for i in self.gt_var:
            frame = i[2]
            if only_bounding_boxes:
                i = i[3:7]
            else:
                del i[2]
            if frame not in gt_var.keys():
                gt_var[frame] = []
            gt_var[frame].append(i)
        self.gt_var = gt_var
