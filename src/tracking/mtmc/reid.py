import os
from collections import defaultdict
import pprint

import numpy as np
import cv2
import torch
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import paired_distances
from tqdm import tqdm
import pickle

from evaluation.idf1 import get_idf1_from_dir
from utils.aicity_reader import parse_annotations_from_txt, group_by_frame, group_by_id, group_in_tracks
from tracking.mtmc.encoder import Encoder
from tracking.mtmc.camera import read_calibration, read_timestamps, angle_to_cam, bbox2gps, time_range, angle

cuda_flag = torch.cuda.is_available()


def generate_file(tracks_by_cam, gen_path="../../../results/reid/S03/"):
    for cam_id, all_tracks_in_camera in tracks_by_cam.items():

        camera_path = os.path.join(gen_path, cam_id)
        if not os.path.exists(camera_path):
            os.makedirs(camera_path)
        filename = os.path.join(camera_path, 'results.txt')

        lines = []
        for track_id, track_object in all_tracks_in_camera.items():
            for det in track_object.track:
                lines.append((det.frame, track_object.id, det.xtl, det.ytl, det.width, det.height, det.score, "-1", "-1", "-1"))

        lines = sorted(lines, key=lambda x: x[0])
        with open(filename, "w") as file:
            for line in lines:
                file.write(",".join(list(map(str, line)))+"\n")


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


def get_track_embeddings(tracks_by_cam, cap, encoder, save_path, batch_size=512):
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
    with open(save_path, 'wb') as f:
        pickle.dump(embeddings, f)

    return embeddings

def get_matches_by_clustering(candidates, embeddings, cam1, id1):
    matches=[]
    clustering = DBSCAN(eps=0.5, min_samples=2)  # TODO: choose appropriate eps
    clustering.fit(np.stack([embeddings[cam][id] for cam, id in [(cam1, id1)] + candidates]))
    label1 = clustering.labels_[0]
    for (cam2, id2), label2 in zip(candidates, clustering.labels_[1:]):
        if label2 == label1:
            matches.append(((cam1, id1), (cam2, id2)))
    print(matches)

def reid_exhaustive(root, save_path):
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
    embeddings = get_track_embeddings(tracks_by_cam, cap, encoder, save_path)
    embeddings = {(cam, id): embd for cam in embeddings for id, embd in embeddings[cam].items()}

    # cluster embeddings to associate tracks
    clustering = DBSCAN(eps=3, min_samples=2)
    clustering.fit(np.stack(list(embeddings.values())))
    clusters = defaultdict(list)
    for id, label in zip(embeddings.keys(), clustering.labels_):
        clusters[label].append(id)
    pprint.pprint(clusters)


def reid_spatiotemporal(root, seq, save_path, metric='euclidean', thresh=10):
    seq_path = os.path.join(root, 'train', seq)
    cams = set([f for f in os.listdir(seq_path) if not f.startswith('.')])

    # read data
    # tracks_by_cam = {cam: group_by_id(parse_annotations_from_txt(os.path.join(seq_path, cam, 'mtsc', 'mtsc_tc_mask_rcnn.txt'))) for cam in cams}
    tracks_by_cam = {cam: group_in_tracks(parse_annotations_from_txt(os.path.join(seq_path, cam, 'gt', 'gt.txt')), cam) for cam in cams}
    cap = {cam: cv2.VideoCapture(os.path.join(seq_path, cam, 'vdo.avi')) for cam in cams}
    fps = {cam: cap[cam].get(cv2.CAP_PROP_FPS) for cam in cams}
    H = {cam: read_calibration(os.path.join(seq_path, cam, 'calibration.txt')) for cam in cams}
    timestamp = read_timestamps(os.path.join(root, 'cam_timestamp', f'{seq}.txt'))
    file_embeddings = os.path.join('../../../results/week5', f'all_embeddings_seq{seq}.pkl')
    load_embeddings = False

    # filter out static tracks
    for cam in cams:
        tracks_by_cam[cam] = dict(filter(lambda x: not is_static(x[1]), tracks_by_cam[cam].items()))

    # initialize encoder
    encoder = Encoder(url='https://drive.google.com/uc?export=download&id=1DH-2KhzOMBmrdslhcavF3UcKu4qbEg9r', n_dims=256, cuda=cuda_flag)
    if cuda_flag:
        encoder = encoder.cuda()
    encoder.eval()

    # load all embeddings
    if load_embeddings:
        with open(file_embeddings, 'rb') as f:
            embeddings = pickle.load(f)
    else:
        # compute all embeddings
        embeddings = get_track_embeddings(tracks_by_cam, cap, encoder, file_embeddings)

    matches = []
    for cam1 in cams:
        for id1, track1 in tracks_by_cam[cam1].items():
            dets1 = track1.get_track()
            dets1.sort(key=lambda det: det.frame)
            dir1 = bbox2gps(dets1[-1].bbox, H[cam1]) - bbox2gps(dets1[-min(int(fps[cam1]), len(dets1)-1)].bbox, H[cam1])
            range1 = time_range(dets1, timestamp[cam1], fps[cam1])
            emb1 = embeddings[cam1][id1]

            min_dist = 100
            match = None
            candidates = []
            for cam2 in cams-{cam1}:
                if angle_to_cam(dets1, H[cam1], cam2) < 45:  # going towards the camera
                    for id2, track2 in tracks_by_cam[cam2].items():
                        dets2 = track2.get_track()
                        dets2.sort(key=lambda det: det.frame)
                        dir2 = bbox2gps(dets2[min(int(fps[cam2]), len(dets2)-1)].bbox, H[cam2]) - bbox2gps(dets2[0].bbox, H[cam2])
                        range2 = time_range(dets2, timestamp[cam2], fps[cam2])

                        if range2[0] >= range1[0]:  # car is detected later in second camera
                            if angle(dir1, dir2) < 15:  # tracks have similar direction
                                if not track2.get_previous_track() and not track1.get_next_track():
                                    # track has not been previously matched to another track from the same direction
                                    candidates.append((cam2, id2)) #TODO Remove when no longer util
                                    emb2 = embeddings[cam2][id2]
                                    dist = paired_distances([emb1], [emb2], metric)[0]
                                    if dist < min_dist:
                                        match = (cam2, track2)
                                        min_dist = dist

            # matches = get_matches_by_clustering(candidates, embeddings, cam1, id1) #TODO Remove when no longer util

            # merge matched tracks
            if match:
                tracks_by_cam[cam1][id1].set_next_track((match[0],match[1].id))
                tracks_by_cam[match[0]][match[1].id].set_previous_track((cam1,id1))

    # search starting tracks (tracks with no prior track)
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
        while(next_track):
            cam_next, id_next = next_track
            track_to_propagate = tracks_by_cam[cam_next][id_next]
            track_to_propagate.id = track_count
            result_reid[track_to_propagate.camera][track_count] = track_to_propagate
            next_track = track_to_propagate.get_next_track()
        track_count+=1

    generate_file(result_reid)


if __name__ == '__main__':
    # reid_exhaustive('../../../data/AIC20_track3/train/S03')
    reid_spatiotemporal('../../../data/AIC20_track3', 'S03', "")

    idf = get_idf1_from_dir("../../../results/reid", "S03", "results", gt_dir='../../../data/AIC20_track3/train')
    print(idf)