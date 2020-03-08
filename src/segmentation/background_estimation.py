from functools import reduce
import numpy as np
import cv2
from tqdm import trange
from src.utils.color_conversion import convert_color


class GaussianModelling:

    def __init__(self, video_path, color_space, reshape_channels=lambda img: img):
        self.cap = cv2.VideoCapture(video_path)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.length = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.color_space = color_space
        self.reshape_channels = reshape_channels


    def fit(self, start=0, length=None):
        if length is None:
            length = self.length
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        # Welford's online variance algorithm
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm

        ret, img = self.cap.read()
        img =  convert_color(img, self.color_space, self.reshape_channels)
        shape = img.shape
        count = np.ones(shape)
        mean = img / count
        M2 = np.zeros(shape)

        for _ in trange(length, desc='modelling background'):
            ret, img = self.cap.read()
            img = convert_color(img, self.color_space, self.reshape_channels)

            count += 1
            delta = img - mean
            mean += delta / count
            delta2 = img - mean
            M2 += delta * delta2

        self.mean = mean
        self.std = np.sqrt(M2 / count)

    def evaluate(self, frame, alpha=10):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, img_bgr = self.cap.read()
        img = convert_color(img_bgr, self.color_space, self.reshape_channels)

        th = alpha * (self.std + 2)

        a = [[img[:,:,i], self.mean[:,:,i], th[:,:,i]] for i in range(img.shape[-1])]
        m = list(map(lambda x: np.abs(x[0] - x[1]) >= x[2], a))
        segmentation = np.array(reduce(lambda a, b: np.bitwise_and(a, b), m))

        return (segmentation * 255).astype(np.uint8), img


if __name__ == '__main__':
    bg_model = GaussianModelling(video_path='../../data/AICity_data/train/S03/c010/vdo.avi', color_space='hsv')
    bg_model.fit(start=0, length=500)

    for frame_id in range(550, 650):
        segmentation, frame = bg_model.evaluate(frame=frame_id, alpha=10)

        cv2.imshow('Frame', cv2.resize(frame, (480, 270)))
        cv2.imshow('Segmentation', cv2.resize(segmentation, (480, 270)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


