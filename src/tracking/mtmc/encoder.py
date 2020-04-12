import os
from collections import defaultdict

import torch
from torch import nn
from torch.hub import load_state_dict_from_url
from torchvision import models
import torchvision.transforms.functional as F

import numpy as np
import cv2
from PIL import Image
from sklearn.metrics.pairwise import paired_distances

from utils.aicity_reader import parse_annotations_from_txt
from tracking.metric_learning.network import EmbeddingNet


class Encoder(nn.Module):
    def __init__(self, url=None):
        super().__init__()
        if url:
            self.model = EmbeddingNet(num_dims=256)
            self.model.load_state_dict(load_state_dict_from_url(url)['state_dict'])
        else:
            self.model = nn.Sequential(
                *list(models.mobilenet_v2(pretrained=True).features.children())[:-1],
                nn.AdaptiveAvgPool2d((1, 1))
            )

    def forward(self, x):
        return self.model(x)

    def get_embedding(self, img):
        with torch.no_grad():
            img = self.transform(img).unsqueeze(0).cuda()
            return self.forward(img).squeeze().cpu().numpy()

    def get_embeddings(self, batch):
        with torch.no_grad():
            batch = torch.stack([self.transform(img) for img in batch]).cuda()
            return self.forward(batch).squeeze().cpu().numpy()

    @staticmethod
    def transform(img):
        img = Image.fromarray(img)
        img = F.resize(img, (80, 100))
        img = F.to_tensor(img)
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

    encoder = Encoder(url='https://drive.google.com/uc?export=download&id=1Op7ABJidoga04SxGCAdcNFBKut1sLY8c')
    print(encoder)
    encoder.cuda()
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
