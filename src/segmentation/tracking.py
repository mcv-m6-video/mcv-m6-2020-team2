from src.utils.track import Track
from src.evaluation.intersection_over_union import bb_intersecion_over_union

def update_tracks(tracks, new_detections, max_track, method):
    if method == 'overlap':
        return update_tracks_by_overlap(tracks, new_detections, max_track)
    elif method == 'kalman':
        return update_tracks_by_overlap(tracks, new_detections, max_track)

def update_tracks_by_overlap(tracks, new_detections, max_track):
    frame_detections = []
    # first frame or empty tracks
    if len(tracks) != 0:
        for detection in new_detections:
            detection.id = max_track + 1
            tracks.append(Track(max_track + 1, [detection]))
            frame_detections.append(detection)
            max_track += 1
    else:
        for track in tracks:
            # Compare track detection in last frame with new detections
            matched_detection = match_next_bbox(track.last_detection(), new_detections)
            # If there's a match, refine detections
            if matched_detection:
                matched_detection.id = track.last_detection().id
                refined_detection = refine_bbox(track.last_detection(), matched_detection)
                track.add_detection(refined_detection)
                frame_detections.append(refined_detection)
                new_detections.remove(matched_detection)

        # Update tracks with unused detections after matching
        for unused_detection in new_detections:
            unused_detection.id = max_track + 1
            tracks.append(Track(max_track + 1, [unused_detection]))
            frame_detections.append(unused_detection)
            max_track += 1

    return tracks, frame_detections

def update_tracks_by_kalman(tracks, new_detections, max_track):
    #TODO
    return tracks, new_detections

def refine_bbox(last_detection, new_detection):
    # TODO Refinement
    return new_detection

def match_next_bbox(last_bbox, unused_detections):
    max_iou = 0
    for detection in unused_detections:
        iou = bb_intersecion_over_union(last_bbox, detection.bbox)
        if iou > max_iou:
            max_iou = iou
            best_match = detection
    if max_iou > 0:
        best_match.id = last_bbox.id
        return best_match
    else:
        return None


def tracking_by_overlap(annotations, frame, iou_th=0.7):
    '''
    OLD NOW NOT USED
    1.-Assign a unique ID to each new detected object in frame N.
    2.-Assign the same ID to the detected object with the highest overlap (IoU) in frame N+1.
    3.-Return to 1.
    '''

    detections = annotations.get(frame, [])
    # If first frame, set ids
    if frame - 1 not in annotations.keys():
        for i in range(len(detections)):
            detections[i].id = i
        return detections
    else:
        old_detections = annotations.get(frame - 1, [])
        old_ids = set()
        for det in detections:
            # Find closest detection in previous frame
            for old_det in old_detections:
                old_ids.add(old_det.id)
                if bb_intersecion_over_union(det.bbox, old_det.bbox) > iou_th:
                    det.id = old_det.id
                    break
            # Set new id when not matched to any previous id
            if det.id == -1:
                det.id = max(old_ids) + 1

    return detections
