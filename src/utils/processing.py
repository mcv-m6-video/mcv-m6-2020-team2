import numpy as np
import cv2
from skimage.measure import label, regionprops

from src.utils.detection import Detection


def denoise(img):
    # Median filter for impulsive noise
    img = cv2.medianBlur(img, ksize=11)
    # # Non-local means denoising (slow!)
    # img = cv2.fastNlMeansDenoising(img, h=11, templateWindowSize=7, searchWindowSize=21)
    # Opening
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((11, 11), np.uint8))

    return img

def fill_holes(img):
    # Closing for connecting close segments
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Floodfill
    img_floodfill = img.copy()
    h, w = img.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(img_floodfill, mask, (0,0), 255)
    img_floodfill_inv = cv2.bitwise_not(img_floodfill)
    img_out = img | img_floodfill_inv

    return img_out

def bounding_boxes(img, frame, min_area=100):
    detections = []
    for region in regionprops(label(img)):
        if region.area >= min_area:
            ytl, xtl, ybr, xbr = region.bbox
            detections.append(Detection(frame, None, 'car', xtl, ytl, xbr, ybr))

    return detections
