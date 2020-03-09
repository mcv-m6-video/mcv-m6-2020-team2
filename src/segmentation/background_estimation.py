import numpy as np
import cv2
from tqdm import trange

from src.utils.color import convert_from_bgr, default_num_channels


class SingleGaussianBackgroundModel:

    def __init__(self, video_path, color_space='gray', channels=None, resize=None):
        self.cap = cv2.VideoCapture(video_path)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.color_space = color_space
        self.channels = channels
        self.resize = resize

        if self.resize is not None:
            self.height = int(self.height * self.resize)
            self.width = int(self.width * self.resize)

    def fit(self, start=0, length=None):
        if length is None:
            length = self.length
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        # Welford's online variance algorithm
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
        count = 0
        num_channels = default_num_channels(self.color_space) if self.channels is None else len(self.channels)
        mean = np.zeros((self.height, self.width, num_channels))
        M2 = np.zeros((self.height, self.width, num_channels))
        for _ in trange(length, desc='modelling background'):
            img = self._read_and_preprocess()
            count += 1
            delta = img - mean
            mean += delta / count
            delta2 = img - mean
            M2 += delta * delta2
        self.mean = mean
        self.std = np.sqrt(M2 / count)

    def evaluate(self, frame, alpha=2.5, rho=0.01, only_update_bg=True):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        img = self._read_and_preprocess()

        # segment foreground
        fg = np.bitwise_and.reduce(np.abs(img - self.mean) >= alpha * (self.std + 2), axis=2)
        bg = ~fg

        # update background model
        if rho > 0:
            if only_update_bg:
                self.mean[bg, :] = rho * img[bg, :] + (1-rho) * self.mean[bg, :]
                self.std[bg, :] = np.sqrt(rho * np.power(img[bg, :] - self.mean[bg, :], 2) + (1-rho) * np.power(self.std[bg, :], 2))
            else:
                self.mean = rho * img + (1-rho) * self.mean
                self.std = np.sqrt(rho * np.power(img - self.mean, 2) + (1-rho) * np.power(self.std, 2))

        return img, (fg * 255).astype(np.uint8)

    def _read_and_preprocess(self):
        ret, img = self.cap.read()
        if self.resize is not None:
            img = cv2.resize(img, None, fx=self.resize, fy=self.resize, interpolation=cv2.INTER_CUBIC)
        img = convert_from_bgr(img, self.color_space, self.channels)
        return img


def get_bg_substractor(method):
    if method == 'MOG':
        backSub = cv2.bgsegm.createBackgroundSubtractorMOG()
    elif method == 'MOG2':
        backSub = cv2.createBackgroundSubtractorMOG2()
    elif method == 'LSBP':
        backSub = cv2.bgsegm.createBackgroundSubtractorLSBP()
    elif method == 'GMG':
        backSub = cv2.bgsegm.createBackgroundSubtractorGMG()
    elif method == 'KNN':
        backSub = cv2.createBackgroundSubtractorKNN()
    elif method == 'GSOC':
        backSub = cv2.bgsegm.createBackgroundSubtractorGSOC()
    elif method == 'CNT':
        backSub = cv2.bgsegm.createBackgroundSubtractorCNT()
    else:
        raise ValueError(f"Unknown background estimation method {method}. Options are [MOG, MOG2, LSBP, GMG, KNN, GSOC, CNT]")
    return backSub


if __name__ == '__main__':
    bg_model = SingleGaussianBackgroundModel(video_path='../../data/AICity_data/train/S03/c010/vdo.avi',
                                             color_space='hsv', channels=(0, 1), resize=0.5)
    bg_model.fit(start=0, length=500)

    for frame in range(550, 650):
        img, mask = bg_model.evaluate(frame=frame)

        #cv2.imshow('frame', img)
        cv2.imshow('foreground', mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
