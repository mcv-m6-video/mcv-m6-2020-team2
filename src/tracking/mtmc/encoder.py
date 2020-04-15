import os
from collections import defaultdict

import torch
from torch import nn
from torchvision import models
import torchvision.transforms.functional as F

import numpy as np
import cv2
from PIL import Image
from sklearn.metrics.pairwise import paired_distances

from utils.aicity_reader import parse_annotations_from_txt


class Encoder(nn.Module):
    def __init__(self, path=None):
        super().__init__()
        self.cuda = torch.cuda.is_available()
        if path:
            self.model = torch.load(path)
        else:
            self.model = nn.Sequential(
                *list(models.mobilenet_v2(pretrained=True).features.children())[:-1],
                nn.AdaptiveAvgPool2d((1, 1))
            )
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.model = self.model.cuda()

    def forward(self, x):
        try:
            return self.model.get_embedding(x)
        except:
            return self.model(x)

    def get_embedding(self, img):
        with torch.no_grad():
            img = self.transform(img).unsqueeze(0)
            if self.cuda:
                img = img.cuda()
            return self.forward(img).squeeze().cpu().numpy()

    def get_embeddings(self, batch):
        with torch.no_grad():
            batch = torch.stack([self.transform(img) for img in batch])
            if self.cuda:
                batch = batch.cuda()
            return self.forward(batch).squeeze().cpu().numpy()

    @staticmethod
    def transform(img):
        img = Image.fromarray(img)
        img = F.resize(img, (128, 128))
        img = F.to_tensor(img)
        img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return img


def test_encoder(metric='euclidean'):
    root = '../../../data/AIC20_track3/train/S03'
    cams = ['c010', 'c011', 'c012', 'c013', 'c014', 'c015']

    detections = {}
    cap = {}
    for cam in cams:
        frame_detections = defaultdict(list)
        for det in parse_annotations_from_txt(os.path.join(root, cam, 'mtsc', 'mtsc_tc_mask_rcnn.txt')):
            if det.height >= 128 and det.width >= 128:
                frame_detections[det.frame].append(det)
        detections[cam] = frame_detections
        cap[cam] = cv2.VideoCapture(os.path.join(root, cam, 'vdo.avi'))

    def random_detection(cam=None, id=None):
        if cam is None:
            cam = np.random.choice(cams)
        if id is None:
            frame = np.random.choice(list(detections[cam].keys()))
            det = np.random.choice(detections[cam][frame])
        else:
            for frame in np.random.permutation(list(detections[cam].keys())):
                found = False
                for det in detections[cam][frame]:
                    if det.id == id:
                        found = True
                        break
                if found:
                    break
            else:
                raise ValueError(f'id {id} not found in cam {cam}')
        cap[cam].set(cv2.CAP_PROP_POS_FRAMES, det.frame)
        ret, img = cap[cam].read()
        img = img[int(det.ytl):int(det.ybr), int(det.xtl):int(det.xbr)]
        return img, (cam, det.id)

    encoder = Encoder(path='../metric_learning/checkpoints/epoch_19__ckpt.pth')
    print(encoder)
    encoder.eval()

    pairs = [(('c010', 15), ('c011', 29)), None]
    for p in pairs:
        if p is not None:
            img1, info1 = random_detection(*p[0])
            img2, info2 = random_detection(*p[1])
        else:
            img1, info1 = random_detection()
            img2, info2 = random_detection()

        embd1 = encoder.get_embedding(img1)
        embd2 = encoder.get_embedding(img2)

        dist = paired_distances([embd1], [embd2], metric).squeeze()
        print(dist)

        cv2.imshow('{}:{}'.format(*info1), img1)
        cv2.imshow('{}:{}'.format(*info2), img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    test_encoder(metric='euclidean')
