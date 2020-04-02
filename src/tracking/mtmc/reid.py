import os
import argparse
from collections import defaultdict
import pprint

import numpy as np
import cv2
from sklearn.cluster import MeanShift
from tqdm import tqdm

from utils.aicity_reader import parse_annotations_from_txt
from tracking.mtmc.encoder import Encoder


def main(args):
    encoder = Encoder()
    encoder = encoder.cuda()
    encoder.eval()

    tracks = {}
    for camera in ['c010', 'c011', 'c012', 'c013', 'c014', 'c015']:
        camera_path = os.path.join(args.root, camera)
        detections = parse_annotations_from_txt(os.path.join(camera_path, 'mtsc', 'mtsc_tc_mask_rcnn.txt'))
        cap = cv2.VideoCapture(os.path.join(camera_path, 'vdo.avi'))

        # group detections by frame
        frame_detections = defaultdict(list)
        for det in detections:
            frame_detections[det.frame].append(det)

        # process detections frame by frame
        track_embeddings = defaultdict(list)
        batch = []
        ids = []
        for frame in tqdm(frame_detections.keys(), desc=f'cam {camera}'):
            # read frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, img = cap.read()

            # crop and resize detections
            for det in frame_detections[frame]:
                if det.width >= args.width and det.height >= args.height:
                    img_cropped = img[int(det.ytl):int(det.ybr), int(det.xtl):int(det.xbr)]
                    if img_cropped.size > 0:
                        img_resized = cv2.resize(img_cropped, (args.width, args.height))
                        batch.append(img_resized)
                        ids.append(det.id)

            # compute embeddings if enough detections in batch
            if len(batch) >= args.batch_size:
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='../../../data/AIC20_track3/train/S03')
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--height', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=512)
    main(parser.parse_args())
