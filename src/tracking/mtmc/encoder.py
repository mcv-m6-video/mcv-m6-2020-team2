import os
from collections import defaultdict

import torch
from torch import nn
from torchvision import models
import torchvision.transforms.functional as F

import numpy as np
import cv2
from sklearn.metrics.pairwise import paired_distances

from utils.aicity_reader import parse_annotations_from_txt


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            *list(models.mobilenet_v2(pretrained=True).features.children())[:-1],
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        return self.model(x)

    def get_embedding(self, img):
        with torch.no_grad():
            img = F.to_tensor(img).unsqueeze(0).cuda()
            return self.forward(img).squeeze().cpu().numpy()

    def get_embeddings(self, batch):
        with torch.no_grad():
            batch = torch.stack([F.to_tensor(img) for img in batch]).cuda()
            return self.forward(batch).squeeze().cpu().numpy()


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

    def random_detection():
        cam = np.random.choice(cams)
        frame = np.random.choice(list(detections[cam].keys()))
        det = np.random.choice(detections[cam][frame])
        cap[cam].set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, img = cap[cam].read()
        img = img[int(det.ytl):int(det.ybr), int(det.xtl):int(det.xbr)]
        img = cv2.resize(img, (128, 128))
        return img

    encoder = Encoder()
    print(encoder)
    encoder.cuda()
    encoder.eval()

    for _ in range(100):
        img1 = random_detection()
        img2 = random_detection()

        embd1 = encoder.get_embedding(img1)
        embd2 = encoder.get_embedding(img2)

        dist = paired_distances([embd1], [embd2], metric).squeeze()
        print(dist)

        cv2.imshow('img1', img1)
        cv2.imshow('img2', img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    test_encoder(metric='cosine')
