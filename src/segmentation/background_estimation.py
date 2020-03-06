import numpy as np
import cv2
from tqdm import trange


class GaussianModelling:

    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.length = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

    def fit(self, start=0, length=None):
        if length is None:
            length = self.length
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        # Welford's online variance algorithm
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
        count = np.zeros((self.height, self.width))
        mean = np.zeros((self.height, self.width))
        M2 = np.zeros((self.height, self.width))
        for _ in trange(length, desc='modelling background'):
            ret, img = self.cap.read()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            count += 1
            delta = img - mean
            mean += delta / count
            delta2 = img - mean
            M2 += delta * delta2
        self.mean = mean
        self.std = np.sqrt(M2 / count)

    def evaluate(self, frame, alpha=10):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, img = self.cap.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        segmentation = np.abs(img - self.mean) >= alpha * (self.std + 2)
        return (segmentation * 255).astype(np.uint8)


if __name__ == '__main__':
    bg_model = GaussianModelling(video_path='../../data/AICity_data/train/S03/c010/vdo.avi')
    bg_model.fit(start=0, length=500)

    for frame in range(550, 650):
        segmentation = bg_model.evaluate(frame=frame, alpha=10)

        cv2.imshow('frame', segmentation)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
