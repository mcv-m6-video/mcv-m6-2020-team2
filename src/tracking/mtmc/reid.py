import os
import pickle
import pprint
from collections import defaultdict

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import paired_distances
from tqdm import tqdm

from evaluation.idf1 import compute_idf1
from tracking.mtmc.camera import read_calibration, read_timestamps, angle_to_cam, bbox2gps, time_range, angle
from tracking.mtmc.encoder import Encoder
from utils.aicity_reader import parse_annotations_from_txt, group_by_frame, group_by_id, group_in_tracks


def is_static(track, thresh=50):
    std = np.std([det.center for det in track.get_track()], axis=0)
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


def get_track_embeddings(tracks_by_cam, cap, encoder, batch_size=512, save_path=None):
    embeddings = defaultdict(dict)
    for cam in tqdm(tracks_by_cam, desc='Computing embeddings', leave=True):
        # process camera detections frame by frame
        detections = [det for track in tracks_by_cam[cam].values() for det in track.track]
        # detections = [det for track in tracks_by_cam[cam].values() for det in track]
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

    # save embeddings
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(embeddings, f)

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
    cams = sorted([f for f in os.listdir(seq_path) if not f.startswith('.')])

    # read data
    # tracks_by_cam = {cam: group_by_id(parse_annotations_from_txt(os.path.join(seq_path, cam, 'mtsc', 'mtsc_tc_mask_rcnn.txt'))) for cam in cams}
    tracks_by_cam = {cam: group_in_tracks(parse_annotations_from_txt(os.path.join(seq_path, cam, 'gt', 'gt.txt')), cam) for cam in cams}
    cap = {cam: cv2.VideoCapture(os.path.join(seq_path, cam, 'vdo.avi')) for cam in cams}
    fps = {cam: cap[cam].get(cv2.CAP_PROP_FPS) for cam in cams}
    H = {cam: read_calibration(os.path.join(seq_path, cam, 'calibration.txt')) for cam in cams}
    timestamp = read_timestamps(os.path.join(root, 'cam_timestamp', f'{seq}.txt'))

    # filter out static tracks
    for cam in cams:
        tracks_by_cam[cam] = dict(filter(lambda x: not is_static(x[1]), tracks_by_cam[cam].items()))

    # initialize encoder
    encoder = Encoder(path='../metric_learning/checkpoints/epoch_19__ckpt.pth')
    encoder = encoder.cuda()
    encoder.eval()

    # compute all embeddings
    embeddings_file = f'./embeddings/{seq}.pkl'
    if os.path.exists(embeddings_file):
        with open(embeddings_file, 'rb') as f:
            embeddings = pickle.load(f)
    else:
        embeddings = get_track_embeddings(tracks_by_cam, cap, encoder, save_path=embeddings_file)

    # matches = []
    for cam1 in cams:
        for id1, track1 in tracks_by_cam[cam1].items():
            dets1 = track1.get_track()
            dets1.sort(key=lambda det: det.frame)
            dir1 = bbox2gps(dets1[-1].bbox, H[cam1]) - bbox2gps(dets1[-min(int(fps[cam1]), len(dets1) - 1)].bbox,
                                                                H[cam1])
            range1 = time_range(dets1, timestamp[cam1], fps[cam1])
            emb1 = embeddings[cam1][id1]

            min_dist = 100
            match = None
            # candidates = []
            for cam2 in cams:
                if cam2 == cam1:
                    continue

                if angle_to_cam(dets1, H[cam1], cam2) < 45:  # going towards the camera
                    for id2, track2 in tracks_by_cam[cam2].items():
                        dets2 = track2.get_track()
                        dets2.sort(key=lambda det: det.frame)
                        dir2 = bbox2gps(dets2[min(int(fps[cam2]), len(dets2) - 1)].bbox, H[cam2]) - bbox2gps(
                            dets2[0].bbox, H[cam2])
                        range2 = time_range(dets2, timestamp[cam2], fps[cam2])

                        if range2[0] >= range1[0]:  # car is detected later in second camera
                            if angle(dir1, dir2) < 15:  # tracks have similar direction
                                if not track2.get_previous_track() and not track1.get_next_track():
                                    # track has not been previously matched to another track from the same direction
                                    # candidates.append((cam2, id2))  # TODO Remove when no longer util
                                    emb2 = embeddings[cam2][id2]
                                    dist = paired_distances([emb1], [emb2], metric).squeeze()
                                    if dist < min_dist:
                                        match = (cam2, track2)
                                        min_dist = dist

            # merge matched tracks
            if match:
                tracks_by_cam[cam1][id1].set_next_track((match[0], match[1].id))
                tracks_by_cam[match[0]][match[1].id].set_previous_track((cam1, id1))

    starting_tracks = []
    for cam, tracks in tracks_by_cam.items():
        for id, track in tracks.items():
            if track.get_next_track() and not track.get_previous_track():
                starting_tracks.append(track)

    # propagate ids through tracks connected to starting tracks
    result_reid = defaultdict(dict)
    track_count = 1
    for track in starting_tracks:
        track.id = track_count
        result_reid[track.camera][track_count] = track
        next_track = track.get_next_track()

        while next_track:
            cam_next, id_next = next_track
            track_to_propagate = tracks_by_cam[cam_next][id_next]
            track_to_propagate.id = track_count
            result_reid[track_to_propagate.camera][track_count] = track_to_propagate
            next_track = track_to_propagate.get_next_track()
        track_count += 1

    return result_reid


def write_results(tracks_by_cam, path):
    for cam, tracks in tracks_by_cam.items():
        lines = []
        for id, track in tracks.items():
            for det in track.track:
                lines.append((det.frame, track.id, int(det.xtl), int(det.ytl), int(det.width), int(det.height),
                              det.score, '-1', '-1', '-1'))
        lines = sorted(lines, key=lambda x: x[0])

        filename = os.path.join(path, cam, 'results.txt')
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as file:
            for line in lines:
                file.write(','.join(list(map(str, line))) + '\n')


if __name__ == '__main__':
    root = '../../../data/AIC20_track3'
    seq = 'S03'

    results = reid_spatiotemporal(root, seq)
    write_results(results, path=os.path.join('results', seq))

    idf1s = []
    for cam in os.listdir(os.path.join(root, 'train', seq)):
        idf1 = compute_idf1(true_path=os.path.join(root, 'train', seq, cam, 'gt', 'gt.txt'),
                            pred_path=os.path.join('results', seq, cam, 'results.txt'))
        idf1s.append(idf1)
    print(f'IDF1: {np.mean(idf1s)}')
