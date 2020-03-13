from src.evaluation.intersection_over_union import bb_intersecion_over_union

def tracking_by_overlap(annotations, frame, iou_th=0.7):
    '''
    1.-Assign a unique ID to each new detected object in frame N.
    2.-Assign the same ID to the detected object with the highest overlap (IoU) in frame N+1.
    3.-Return to 1.
    '''

    detections = annotations.get(frame, [])
    # If first frame, set ids
    if frame-1 not in annotations.keys():
        for i in range(len(detections)):
            detections[i].id = i
        return detections
    else:
        old_detections = annotations.get(frame-1,[])
        old_ids=set()
        for det in detections:
            # Find closest detection in previous frame
            for old_det in old_detections:
                old_ids.add(old_det.id)
                if bb_intersecion_over_union(det.bbox, old_det.bbox) > iou_th:
                    det.id = old_det.id
                    break
            # Set new id when not matched to any previous id
            if det.id == -1:
                det.id = max(old_ids)+1

    return detections