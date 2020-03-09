import os
from collections import defaultdict

import numpy as np
import cv2
from tqdm import trange
import matplotlib.pyplot as plt
import imageio

from src.utils.aicity_reader import AICityChallengeAnnotationReader
from src.segmentation.background_estimation import SingleGaussianBackgroundModel, get_bg_substractor
from src.utils.detection import Detection
from src.evaluation.average_precision import mean_average_precision
from src.utils.processing import denoise, fill_holes, bounding_boxes


def task1_2(adaptive, random_search, model_frac=0.25, min_width=120, max_width=800, min_height=100, max_height=600,
            debug=0, save_path=None):
    reader = AICityChallengeAnnotationReader(path='data/AICity_data/train/S03/c010/gt/gt.txt')
    gt = reader.get_annotations(classes=['car'], only_not_parked=True)

    bg_model = SingleGaussianBackgroundModel(video_path='data/AICity_data/train/S03/c010/vdo.avi')
    video_length = bg_model.length
    bg_model.fit(start=0, length=int(video_length * model_frac))

    roi = cv2.imread('data/AICity_data/train/S03/c010/roi.jpg', cv2.IMREAD_GRAYSCALE)

    start_frame = int(video_length * model_frac)
    end_frame = video_length

    # hyperparameter search
    if random_search:
        alphas = np.random.choice(np.linspace(2, 4, 50), 25)
        rhos = np.random.choice(np.linspace(0.001, 0.1, 50), 25) if adaptive else [0]
        combinations = [(alpha, rho) for alpha, rho in zip(alphas, rhos)]
    else:
        alphas = [2, 2.5, 3, 3.5, 4]
        rhos = [0.005, 0.01, 0.025, 0.05, 0.1] if adaptive else [0]
        combinations = [(alpha, rho) for alpha in alphas for rho in rhos]

    for alpha, rho in combinations:
        if save_path:
            writer = imageio.get_writer(os.path.join(save_path, f'task1_2_alpha{alpha:.1f}_rho{rho:.3f}.gif'), fps=10)

        y_true = []
        y_pred = []
        for frame in trange(start_frame, end_frame, desc=f'evaluating frames'):
            _, mask = bg_model.evaluate(frame=frame, alpha=alpha, rho=rho)
            mask = mask & roi
            if debug >= 2:
                plt.imshow(mask); plt.show()

            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            if debug >= 2:
                plt.imshow(mask); plt.show()

            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            detections = []
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                if min_width < w < max_width and min_height < h < max_height:
                    detections.append(Detection(frame, None, 'car', x, y, x + w, y + h))
            annotations = gt.get(frame, [])

            if debug >= 1 or save_path:
                img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                for det in detections:
                    cv2.rectangle(img, (det.xtl, det.ytl), (det.xbr, det.ybr), (0, 255, 0), 2)
                for det in annotations:
                    cv2.rectangle(img, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ybr)), (0, 0, 255), 2)

                if save_path:
                    writer.append_data(img)

                if debug >= 1:
                    cv2.imshow('result', img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            y_pred.append(detections)
            y_true.append(annotations)

        if save_path:
            writer.close()

        ap = mean_average_precision(y_true, y_pred, classes=['car'])
        print(f'alpha: {alpha:.1f}, rho: {rho:.3f}, AP: {ap:.4f}')


def task1(debug=0, save_path=None):
    """
    Gaussian modelling
    """

    task1_2(False, False, debug=debug, save_path=save_path)


def task2(debug=0, save_path=None):
    """
    Adaptive modelling
    """

    task1_2(True, True, debug=debug, save_path=save_path)


def task3(bg_subst_methods, model_frac=0.25, history=10, debug=0):
    """
    Comparison with the state of the art
    """

    # Load ground truth
    reader = AICityChallengeAnnotationReader(path='data/AICity_data/train/S03/c010/gt/gt.txt')
    gt = reader.get_annotations(classes=['car'], only_not_parked=True)

    cap = cv2.VideoCapture('data/AICity_data/train/S03/c010/vdo.avi')
    video_length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    start_frame = int(video_length * model_frac)
    end_frame = int(video_length)

    y_pred = defaultdict(list)
    y_true = defaultdict(list)
    ap = defaultdict(list)

    for sota_method in bg_subst_methods:
        backSub = get_bg_substractor(sota_method)
        for frame in trange(start_frame, end_frame, desc='evaluating frames'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            retVal, img = cap.read()

            mask = backSub.apply(img, learningRate=1.0 / history)
            if debug >= 2:
                cv2.imshow('Foreground', cv2.resize(mask, (960, 540)))
                cv2.imshow('Original', cv2.resize(img, (960, 540)))

            detections = bounding_boxes(mask, frame=frame, min_height=100, max_height=600, min_width=120, max_width=800)
            annotations = gt.get(frame, [])
            y_pred[sota_method].append(detections)
            y_true[sota_method].append(annotations)

            if debug >= 1:
                for det in detections:
                    cv2.rectangle(img, (det.xtl, det.ytl), (det.xbr, det.ybr), (0, 255, 0), 2)
                for det in annotations:
                    cv2.rectangle(img, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ybr)), (0, 0, 255), 2)
                cv2.imshow('frame', img)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        ap[sota_method] = mean_average_precision(y_true[sota_method], y_pred[sota_method], classes=['car'])
        print('Method:', sota_method, f'AP: {ap[sota_method]:.4f}')


def task4():
    """
    Color modelling
    """

    alpha = 2
    shape = (480, 270)
    color_space = 'gray'
    channels = [True, True, True]

    reader = AICityChallengeAnnotationReader(path='data/AICity_data/train/S03/c010/gt/gt.txt')
    gt = reader.get_annotations(classes=['car'], only_not_parked=True)

    bg_model = SingleGaussianBackgroundModel(video_path='data/AICity_data/train/S03/c010/vdo.avi',
                                             color_space=color_space, channels=channels)
    video_length = bg_model.length
    bg_model.fit(start=0, length=int(video_length * 0.25))

    y_true = []
    y_pred = []

    for frame in trange(int(video_length * 0.25), video_length, desc='obtaining foreground and detecting objects'):
        frame_img, segmentation = bg_model.evaluate(frame=frame, alpha=alpha)
        segmentation_denoised = denoise(segmentation)
        segmentation_filled = fill_holes(segmentation_denoised)

        y_pred.append(
            bounding_boxes(segmentation, frame=frame, min_height=100, max_height=600, min_width=120, max_width=800))
        if frame in gt.keys():
            y_true.append(gt[frame])
        else:
            y_true.append([])

        for item_pred, item_true in zip(y_pred[-1], y_true[-1]):
            cv2.rectangle(frame_img,
                          (int(item_pred.xtl), int(item_pred.ytl)),
                          (int(item_pred.xbr), int(item_pred.ybr)),
                          (0, 0, 255), 4)
            cv2.rectangle(frame_img,
                          (int(item_true.xtl), int(item_true.ytl)),
                          (int(item_true.xbr), int(item_true.ybr)),
                          (0, 255, 0), 2)
        cv2.imshow(f'BGR Image', cv2.resize(frame_img, shape))

        # show the channels we are using
        # for i in range(image_channels.shape[-1]):
        #     cv2.imshow(f'Channels_{i}', cv2.resize(image_channels[:,:,i], shape))
        cv2.imshow(f'Segmentation using {color_space}', cv2.resize(segmentation, shape))
        cv2.imshow(f'Segmentation Morphed using {color_space}', cv2.resize(segmentation_filled, shape))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    ap = mean_average_precision(y_true, y_pred, classes=['car'])
    print(f'AP: {ap:.4f}')


if __name__ == '__main__':
    #task1(save_path='results/week2')
    task2(save_path='results/week2')
    #task3(["MOG", "MOG2", "LSBP", "GMG", "KNN", "GSOC", "CNT"], debug=1)
    #task4()
