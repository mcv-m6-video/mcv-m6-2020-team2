import cv2

from src.utils.detection import Detection


def postprocess(mask):
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    return mask


def bounding_boxes(mask, min_height, max_height, min_width, max_width, frame):
    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if min_width < w < max_width and min_height < h < max_height:
            detections.append(Detection(frame, None, 'car', x, y, x + w, y + h))
    return detections
