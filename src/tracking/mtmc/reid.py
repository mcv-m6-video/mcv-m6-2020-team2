import os
from collections import defaultdict
import pprint

import numpy as np
import cv2
from sklearn.cluster import MeanShift
from sklearn.metrics.pairwise import paired_distances
from tqdm import tqdm

from utils.aicity_reader import parse_annotations_from_txt, group_by_frame, group_by_id
from tracking.mtmc.encoder import Encoder
from tracking.mtmc.camera import read_calibration, read_timestamps, angle_to_cam, bbox2gps, time_range, angle


def reid_exhaustive(root, width, height, batch_size):
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


def reid_spatiotemporal(root, seq, metric='euclidean', thresh=0.1):
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

    matches = []
    for cam1 in cams:
        pbar1 = tqdm(tracks_by_cam[cam1].items(), desc=f'cam1={cam1}', leave=True)
        for id1, track1 in pbar1:
            pbar1.set_postfix({'id': id1})

            track1.sort(key=lambda det: det.frame)
            dir1 = bbox2gps(track1[-1].bbox, H[cam1]) - bbox2gps(track1[-min(int(fps[cam1]), len(track1)-1)].bbox, H[cam1])
            range1 = time_range(track1, timestamp[cam1], fps[cam1])
            embd1 = None

            for cam2 in cams-{cam1}:
                if angle_to_cam(track1, H[cam1], cam2) < 45:  # going towards the camera
                    pbar2 = tqdm(tracks_by_cam[cam2].items(), desc=f'cam2={cam2}', leave=False)
                    for id2, track2 in pbar2:
                        pbar2.set_postfix({'id': id2})

                        track2.sort(key=lambda det: det.frame)
                        dir2 = bbox2gps(track2[min(int(fps[cam2]), len(track2)-1)].bbox, H[cam2]) - bbox2gps(track2[0].bbox, H[cam2])
                        range2 = time_range(track2, timestamp[cam2], fps[cam2])

                        if range2[0] >= range1[0]:  # car is detected later in second camera
                            if angle(dir1, dir2) < 15:  # tracks have similar direction
                                if embd1 is None:
                                    embd1 = get_track_embedding(track1, cap[cam1], encoder)
                                embd2 = get_track_embedding(track2, cap[cam2], encoder)
                                dist = paired_distances([embd1], [embd2], metric)
                                if dist < thresh:  # TODO: improve this
                                    matches.append(((cam1, id1), (cam2, id2)))

    print(matches)


if __name__ == '__main__':
    # reid_exhaustive('../../../data/AIC20_track3/train/S03', width=128, height=128, batch_size=512)
    reid_spatiotemporal('../../../data/AIC20_track3', 'S03')
