import numpy as np
import cv2
from tqdm import trange

from src.utils.color import convert_bgr, num_channels


class SingleGaussianBackgroundModel:

    def __init__(self, video_path, color_space='gray'):
        self.cap = cv2.VideoCapture(video_path)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.color_space = color_space
        self.channels = num_channels(self.color_space)

    def fit(self, start=0, length=None):
        if length is None:
            length = self.length
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        # Welford's online variance algorithm
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
        count = 0
        mean = np.zeros((self.height, self.width, self.channels))
        M2 = np.zeros((self.height, self.width, self.channels))
        for _ in trange(length, desc='modelling background'):
            ret, img = self.cap.read()
            img = convert_bgr(img, self.color_space)
            count += 1
            delta = img - mean
            mean += delta / count
            delta2 = img - mean
            M2 += delta * delta2
        self.mean = mean
        self.std = np.sqrt(M2 / count)

    def evaluate(self, frame, alpha=2.5, rho=0.01, only_update_bg=True):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, img = self.cap.read()
        img = convert_bgr(img, self.color_space)

        # segment foreground
        fg = np.bitwise_and.reduce(np.abs(img - self.mean) >= alpha * (self.std + 2), axis=2)
        bg = ~fg

        # update background model
        if only_update_bg:
            self.mean[bg, :] = rho * img[bg, :] + (1-rho) * self.mean[bg, :]
            self.std[bg, :] = np.sqrt(rho * np.power(img[bg, :] - self.mean[bg, :], 2) + (1-rho) * np.power(self.std[bg, :], 2))
        else:
            self.mean = rho * img + (1-rho) * self.mean
            self.std = np.sqrt(rho * np.power(img - self.mean, 2) + (1-rho) * np.power(self.std, 2))

        return img, (fg * 255).astype(np.uint8)


if __name__ == '__main__':
    bg_model = SingleGaussianBackgroundModel(video_path='../../data/AICity_data/train/S03/c010/vdo.avi')
    bg_model.fit(start=0, length=500)

    for frame in range(550, 650):
        img, mask = bg_model.evaluate(frame=frame)

        cv2.imshow('frame', img)
        cv2.imshow('foreground', mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
