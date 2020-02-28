def IoU_from_bbs(bb1, bb2):
    """ Calculate the Intersection over Union (IoU) of two bounding boxes. """

    # Determine intersection rectangle coordinates
    left = max(bb1[0], bb2[0])
    top = max(bb1[1], bb2[1])
    right = min(bb1[2], bb2[2])
    bottom = min(bb1[3], bb2[3])

    # Check non-overlapping rectangle
    if right < left or bottom < top:
        return 0

    # Compute intersection area
    intersection_area = (right - left) * (bottom - top)

    # Compute area of bbs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # Compute IoU
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    return iou
