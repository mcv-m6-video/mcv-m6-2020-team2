import os
from collections import defaultdict
import pprint

import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from utils.aicity_reader import parse_annotations_from_txt, group_by_frame, group_by_id
from tracking.mtmc.encoder import Encoder
from tracking.mtmc.camera import read_calibration, read_timestamps, angle_to_cam, bbox2gps, time_range, angle


def is_static(track, thresh=50):
    std = np.std([det.center for det in track], axis=0)
    return np.all(std < thresh)


def get_track_embedding(track, cap, encoder, max_views=32):
    batch = []
    for det in np.random.permutation(track)[:max_views]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, det.frame)
        ret, img = cap.read()
        img = img[int(det.ytl):int(det.ybr), int(det.xtl):int(det.xbr)]
        if img.size > 0:
            batch.append(img)

    embeddings = encoder.get_embeddings(batch)

    # combine track embeddings by averaging them
    embedding = embeddings.mean(axis=0)

    return embedding


def get_track_embeddings(tracks_by_cam, cap, encoder, batch_size=512):
    embeddings = defaultdict(dict)
    for cam in tqdm(tracks_by_cam, desc='Computing embeddings', leave=True):
        # process camera detections frame by frame
        detections = [det for track in tracks_by_cam[cam].values() for det in track]
        detections_by_frame = group_by_frame(detections)

        cap[cam].set(cv2.CAP_PROP_POS_FRAMES, 0)
        length = int(cap[cam].get(cv2.CAP_PROP_FRAME_COUNT))

        track_embeddings = defaultdict(list)
        batch = []
        ids = []
        for _ in tqdm(range(length), desc=f'cam={cam}', leave=False):
            # read frame
            frame = int(cap[cam].get(cv2.CAP_PROP_POS_FRAMES))
            _, img = cap[cam].read()
            if frame not in detections_by_frame:
                continue

            # crop and accumulate frame detections
            for det in detections_by_frame[frame]:
                crop = img[int(det.ytl):int(det.ybr), int(det.xtl):int(det.xbr)]
                if crop.size > 0:
                    batch.append(crop)
                    ids.append(det.id)

            # compute embeddings if enough detections in batch
            if len(batch) >= batch_size:
                embds = encoder.get_embeddings(batch)
                for id, embd in zip(ids, embds):
                    track_embeddings[id].append(embd)
                batch.clear()
                ids.clear()

        # compute embeddings of last batch
        if len(batch) > 0:
            embds = encoder.get_embeddings(batch)
            for id, embd in zip(ids, embds):
                track_embeddings[id].append(embd)

        # combine track embeddings by averaging them
        for id, embds in track_embeddings.items():
            embeddings[cam][id] = np.stack(embds).mean(axis=0)

    return embeddings


def reid_exhaustive(root):
    cams = set(os.listdir(root))

    # read data
    tracks_by_cam = {cam: group_by_id(parse_annotations_from_txt(os.path.join(root, cam, 'mtsc', 'mtsc_tc_mask_rcnn.txt'))) for cam in cams}
    cap = {cam: cv2.VideoCapture(os.path.join(root, cam, 'vdo.avi')) for cam in cams}

    # filter out static tracks
    for cam in cams:
        tracks_by_cam[cam] = dict(filter(lambda x: not is_static(x[1]), tracks_by_cam[cam].items()))

    # initialize encoder
    encoder = Encoder()
    encoder = encoder.cuda()
    encoder.eval()

    # compute all embeddings
    embeddings = get_track_embeddings(tracks_by_cam, cap, encoder)
    embeddings = {(cam, id): embd for cam in embeddings for id, embd in embeddings[cam].items()}

    # cluster embeddings to associate tracks
    clustering = DBSCAN(eps=3, min_samples=2)
    clustering.fit(np.stack(list(embeddings.values())))
    clusters = defaultdict(list)
    for id, label in zip(embeddings.keys(), clustering.labels_):
        clusters[label].append(id)
    pprint.pprint(clusters)


def reid_spatiotemporal(root, seq, metric='euclidean', thresh=10):
    seq_path = os.path.join(root, 'train', seq)
    cams = set(os.listdir(seq_path))

    # read data
    tracks_by_cam = {cam: group_by_id(parse_annotations_from_txt(os.path.join(seq_path, cam, 'mtsc', 'mtsc_tc_mask_rcnn.txt'))) for cam in cams}
    cap = {cam: cv2.VideoCapture(os.path.join(seq_path, cam, 'vdo.avi')) for cam in cams}
    fps = {cam: cap[cam].get(cv2.CAP_PROP_FPS) for cam in cams}
    H = {cam: read_calibration(os.path.join(seq_path, cam, 'calibration.txt')) for cam in cams}
    timestamp = read_timestamps(os.path.join(root, 'cam_timestamp', f'{seq}.txt'))

    # filter out static tracks
    for cam in cams:
        tracks_by_cam[cam] = dict(filter(lambda x: not is_static(x[1]), tracks_by_cam[cam].items()))

    # initialize encoder
    encoder = Encoder()
    encoder = encoder.cuda()
    encoder.eval()

    # compute all embeddings
    embeddings = get_track_embeddings(tracks_by_cam, cap, encoder)

    matches = []
    for cam1 in cams:
        for id1, track1 in tracks_by_cam[cam1].items():
            track1.sort(key=lambda det: det.frame)
            dir1 = bbox2gps(track1[-1].bbox, H[cam1]) - bbox2gps(track1[-min(int(fps[cam1]), len(track1)-1)].bbox, H[cam1])
            range1 = time_range(track1, timestamp[cam1], fps[cam1])

            candidates = []
            for cam2 in cams-{cam1}:
                if angle_to_cam(track1, H[cam1], cam2) < 45:  # going towards the camera
                    for id2, track2 in tracks_by_cam[cam2].items():
                        track2.sort(key=lambda det: det.frame)
                        dir2 = bbox2gps(track2[min(int(fps[cam2]), len(track2)-1)].bbox, H[cam2]) - bbox2gps(track2[0].bbox, H[cam2])
                        range2 = time_range(track2, timestamp[cam2], fps[cam2])

                        if range2[0] >= range1[0]:  # car is detected later in second camera
                            if angle(dir1, dir2) < 15:  # tracks have similar direction
                                candidates.append((cam2, id2))

            clustering = DBSCAN(eps=3, min_samples=2)  # TODO: choose appropriate eps
            clustering.fit(np.stack([embeddings[cam][id] for cam, id in [(cam1, id1)] + candidates]))
            label1 = clustering.labels_[0]
            for (cam2, id2), label2 in zip(candidates, clustering.labels_[1:]):
                if label2 == label1:
                    matches.append(((cam1, id1), (cam2, id2)))
    print(matches)


if __name__ == '__main__':
    # reid_exhaustive('../../../data/AIC20_track3/train/S03')
    reid_spatiotemporal('../../../data/AIC20_track3', 'S03')
