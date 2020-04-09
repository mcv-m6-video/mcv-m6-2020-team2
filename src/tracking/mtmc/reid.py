import os
import argparse
from collections import defaultdict
import pprint

import numpy as np
import cv2
from sklearn.cluster import MeanShift
from tqdm import tqdm

from utils.aicity_reader import parse_annotations_from_txt, group_by_frame, group_by_id
from tracking.mtmc.encoder import Encoder
from tracking.mtmc.camera import read_calibration, read_timestamps, project_bbox
from evaluation.intersection_over_union import vec_intersecion_over_union


def reid(root, width, height, batch_size):
    encoder = Encoder()
    encoder = encoder.cuda()
    encoder.eval()

    tracks = {}
    for camera in ['c010', 'c011', 'c012', 'c013', 'c014', 'c015']:
        camera_path = os.path.join(root, camera)
        detections = group_by_frame(parse_annotations_from_txt(os.path.join(camera_path, 'mtsc', 'mtsc_tc_mask_rcnn.txt')))
        cap = cv2.VideoCapture(os.path.join(camera_path, 'vdo.avi'))

        # process detections frame by frame
        track_embeddings = defaultdict(list)
        batch = []
        ids = []
        for frame in tqdm(detections.keys(), desc=f'cam {camera}'):
            # read frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, img = cap.read()

            # crop and resize detections
            for det in detections[frame]:
                if det.width >= width and det.height >= height:
                    img_cropped = img[int(det.ytl):int(det.ybr), int(det.xtl):int(det.xbr)]
                    if img_cropped.size > 0:
                        img_resized = cv2.resize(img_cropped, (width, height))
                        batch.append(img_resized)
                        ids.append(det.id)

            # compute embeddings if enough detections in batch
            if len(batch) >= batch_size:
                embeddings = encoder.get_embeddings(batch)
                for id, embd in zip(ids, embeddings):
                    track_embeddings[id].append(embd)
                batch.clear()
                ids.clear()

        # compute embeddings of last batch
        if len(batch) > 0:
            embeddings = encoder.get_embeddings(batch)
            for id, embd in zip(ids, embeddings):
                track_embeddings[id].append(embd)

        # combine embeddings of each track
        for id, embeddings in track_embeddings.items():
            tracks[(camera, id)] = np.stack(embeddings).mean(axis=0)

    # cluster embeddings to associate tracks
    ids = list(tracks.keys())
    embeddings = list(tracks.values())
    ms = MeanShift()
    ms.fit(np.stack(embeddings))
    clusters = defaultdict(list)
    for id, label in zip(ids, ms.labels_):
        clusters[label].append(id)
    pprint.pprint(clusters)


def candidates_by_trajectory(root, cam1, cam2, thresh=0.2):
    dets = {cam: group_by_id(parse_annotations_from_txt(os.path.join(root, cam, 'mtsc', 'mtsc_tc_mask_rcnn.txt'))) for cam in [cam1, cam2]}
    H = {cam: read_calibration(os.path.join(root, cam, 'calibration.txt')) for cam in [cam1, cam2]}

    matches = []
    for id1, track1 in tqdm(dets[cam1].items()):
        for id2, track2 in dets[cam2].items():
            # warp boxes from cam2 to cam1
            boxes1 = np.array([det.bbox for det in track1])
            boxes2 = np.array([project_bbox(det.bbox, H[cam2], H[cam1]) for det in track2])
            iou = vec_intersecion_over_union(boxes1, boxes2)

            nz = np.count_nonzero(iou)
            if nz > 0:
                miou = np.sum(iou) / nz
                if miou > thresh:
                    matches.append((id1, id2, miou))
    return sorted(matches, key=lambda x: x[2], reverse=True)


def candidates_by_timestamp(root, seq, cam1, cam2):
    dets = {cam: group_by_id(parse_annotations_from_txt(os.path.join(root, 'train', seq, cam, 'mtsc', 'mtsc_tc_mask_rcnn.txt'))) for cam in [cam1, cam2]}
    cap = {cam: cv2.VideoCapture(os.path.join(root, 'train', seq, cam, 'vdo.avi')) for cam in [cam1, cam2]}
    fps = {cam: cap[cam].get(cv2.CAP_PROP_FPS) for cam in [cam1, cam2]}
    timestamp = read_timestamps(os.path.join(root, 'cam_timestamp', f'{seq}.txt'))

    matches = []
    for id1, track1 in tqdm(dets[cam1].items()):
        timestamps1 = sorted([timestamp[cam1] + det.frame / fps[cam1] for det in track1])
        s1 = timestamps1[0]
        e1 = timestamps1[-1]
        for id2, track2 in dets[cam2].items():
            timestamps2 = sorted([timestamp[cam2] + det.frame / fps[cam2] for det in track2])
            s2 = timestamps2[0]
            e2 = timestamps2[-1]

            # tracks overlap in time
            if (s1 <= e2) and (s2 <= e1):
                inter = min(e1, e2) - max(s1, s2)
                matches.append((id1, id2, inter))
    return sorted(matches, key=lambda x: x[2], reverse=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='../../../data/AIC20_track3/train/S03')
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--height', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=512)
    args = parser.parse_args()
    reid(args.root, args.width, args.height, args.batch_size)
