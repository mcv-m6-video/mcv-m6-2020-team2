from copy import deepcopy
import numpy as np

from src.utils.detection import Detection
from src.utils.track import Track
from src.evaluation.intersection_over_union import bb_intersecion_over_union


def update_tracks_by_overlap(tracks, new_detections, max_track, refinement=True, optical_flow=None):
    new_detections_copy = deepcopy(new_detections)
    frame_tracks = []
    for track in tracks:
        # Compare track detection in last frame with new detections
        matched_detection = match_next_bbox(track.last_detection(), new_detections_copy, optical_flow)
        # If there's a match, refine detections
        if matched_detection:
            if refinement:
                refined_detection = refine_bbox(track.get_track()[-2:], matched_detection)
            else:
                refined_detection = deepcopy(matched_detection)
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

def refine_bbox(last_detections, new_detection, k=0.5):
    # No refinement for the first two frames
    if len(last_detections) < 2:
        return new_detection

    # Predict coordinates of new detection from last two detections
    pred_detection_xtl = 2 * last_detections[1].xtl - last_detections[0].xtl
    pred_detection_ytl = 2 * last_detections[1].ytl - last_detections[0].ytl
    pred_detection_xbr = 2 * last_detections[1].xbr - last_detections[0].xbr
    pred_detection_ybr = 2 * last_detections[1].ybr - last_detections[0].ybr

    # Compute average of predicted coordinates and detected coordinates
    refined_xtl = new_detection.xtl * k + pred_detection_xtl * (1 - k)
    refined_ytl = new_detection.ytl * k + pred_detection_ytl * (1 - k)
    refined_xbr = new_detection.xbr * k + pred_detection_xbr * (1 - k)
    refined_ybr = new_detection.ybr * k + pred_detection_ybr * (1 - k)

    # Get refined detection
    refined_detection = Detection(frame=new_detection.frame,
                                  id=new_detection.id,
                                  label=new_detection.label,
                                  xtl=refined_xtl,
                                  ytl=refined_ytl,
                                  xbr=refined_xbr,
                                  ybr=refined_ybr,
                                  score=new_detection.score)

    return refined_detection

def match_next_bbox(last_detection, unused_detections, optical_flow):
    last_detection_copy = deepcopy(last_detection)

    # Compensate last_detection
    if optical_flow is not None:
        last_detection_copy.xtl += optical_flow[int(last_detection_copy.ytl), int(last_detection_copy.xtl), 0]
        last_detection_copy.ytl += optical_flow[int(last_detection_copy.ytl), int(last_detection_copy.xtl), 1]
        last_detection_copy.xbr += optical_flow[int(last_detection_copy.ybr), int(last_detection_copy.xbr), 0]
        last_detection_copy.ybr += optical_flow[int(last_detection_copy.ybr), int(last_detection_copy.xbr), 1]

    max_iou = 0
    for detection in unused_detections:
        iou = bb_intersecion_over_union(last_detection_copy.bbox, detection.bbox)
        if iou > max_iou:
            max_iou = iou
            best_match = detection
    if max_iou > 0:
        best_match.id = last_detection_copy.id
        return best_match
    else:
        return None
