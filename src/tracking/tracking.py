from copy import deepcopy
import numpy as np

from src.utils.detection import Detection
from src.utils.track import Track
from src.evaluation.intersection_over_union import bb_intersecion_over_union

def update_tracks_by_overlap(tracks, new_detections, max_track):
    new_detections_copy = deepcopy(new_detections)
    frame_tracks = []
    for track in tracks:
        # Compare track detection in last frame with new detections
        matched_detection = match_next_bbox(track.last_detection(), new_detections_copy)
        # If there's a match, refine detections
        if matched_detection:
            refined_detection = refine_bbox(track.get_track()[-2:], matched_detection)
            track.add_detection(refined_detection)
            frame_tracks.append(track)
            new_detections_copy.remove(matched_detection)

    # Update tracks with unused detections after matching
    for unused_detection in new_detections_copy:
        unused_detection.id = max_track + 1
        new_track = Track(max_track + 1, [unused_detection])
        tracks.append(new_track)
        frame_tracks.append(new_track)
        max_track += 1

    return tracks, frame_tracks, max_track

def refine_bbox(last_detections, new_detection):
    # No refinement for the first two frames
    if len(last_detections) < 2:
        return new_detection

    # Compute centroids of last two detections and new detection
    last_detections_c = np.array([((i.xtl+i.xbr)/2, (i.ytl+i.ybr)/2) for i in last_detections])
    new_detection_c = np.array([(new_detection.xtl+new_detection.xbr)/2, (new_detection.ytl+new_detection.ybr)/2])

    # Predict centroid of new detection from last two detections
    pred_detection_c = 2 * last_detections_c[1] - last_detections_c[0]

    # Compute average of predicted centroid and detected centroid
    refined_c = (new_detection_c + pred_detection_c) / 2

    # Compute refined detection
    w = new_detection.xbr - new_detection.xtl
    h = new_detection.ybr - new_detection.ytl

    refined_detection = Detection(frame=new_detection.frame,
                                  id=new_detection.id,
                                  label=new_detection.label,
                                  xtl=refined_c[0] - w/2,
                                  ytl=refined_c[1] - h/2,
                                  xbr=refined_c[0] + w/2,
                                  ybr=refined_c[1] + h/2,
                                  score=new_detection.score)

    return refined_detection

def match_next_bbox(last_detection, unused_detections):
    max_iou = 0
    for detection in unused_detections:
        iou = bb_intersecion_over_union(last_detection.bbox, detection.bbox)
        if iou > max_iou:
            max_iou = iou
            best_match = detection
    if max_iou > 0:
        best_match.id = last_detection.id
        return best_match
    else:
        return None