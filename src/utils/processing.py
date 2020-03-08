import numpy as np
import cv2
from skimage.measure import label, regionprops

from src.utils.detection import Detection


def denoise(img):
    # Median filter for impulsive noise
    img = cv2.medianBlur(img, ksize=3)
    # Opening
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))

    return img

def fill_holes(img):
    # Floodfill
    img_floodfill = img.copy()
    h, w = img.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(img_floodfill, mask, (0,0), 255)
    img_floodfill_inv = cv2.bitwise_not(img_floodfill)
    img_out = img | img_floodfill_inv

    return img_out

def bounding_boxes(img, frame, min_height, max_height, min_width, max_width):
    detections = []
    for region in regionprops(label(img)):
        ytl, xtl, ybr, xbr = region.bbox
        if max_width >= ybr - ytl >= min_width and max_height >= xbr - xtl >= min_height:
            detections.append(Detection(frame, None, 'car', xtl, ytl, xbr, ybr))

    return detections
