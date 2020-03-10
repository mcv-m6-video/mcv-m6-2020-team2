import os

import numpy as np
import cv2
from tqdm import trange
import imageio
import time

from src.utils.aicity_reader import AICityChallengeAnnotationReader
from src.segmentation.background_estimation import SingleGaussianBackgroundModel, sota_bg_subtractor
from src.utils.processing import postprocess, bounding_boxes
from src.evaluation.average_precision import mean_average_precision


def task1_2(adaptive, random_search, model_frac=0.25, min_width=120, max_width=800, min_height=100, max_height=600,
            debug=0, save_path=None):
    reader = AICityChallengeAnnotationReader(path='data/AICity_data/train/S03/c010/gt/gt.txt')
    gt = reader.get_annotations(classes=['car'], only_not_parked=True)

    roi = cv2.imread('data/AICity_data/train/S03/c010/roi.jpg', cv2.IMREAD_GRAYSCALE)

    bg_model = SingleGaussianBackgroundModel(video_path='data/AICity_data/train/S03/c010/vdo.avi')
    video_length = bg_model.length
    bg_model.fit(start=0, length=int(video_length * model_frac))

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
        for frame in trange(start_frame, end_frame, desc='evaluating frames'):
            _, mask, _ = bg_model.evaluate(frame=frame, alpha=alpha, rho=rho)
            mask = mask & roi
            mask = postprocess(mask)

            detections = bounding_boxes(mask, min_height, max_height, min_width, max_width, frame)

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

        cv2.destroyAllWindows()

        if save_path:
            writer.close()

        ap, prec, rec = mean_average_precision(y_true, y_pred, classes=['car'])
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


def task3(methods, model_frac=0.25, min_width=120, max_width=800, min_height=100, max_height=600, save_path=None, debug=0):
    """
    Comparison with the state of the art
    """

    reader = AICityChallengeAnnotationReader(path='data/AICity_data/train/S03/c010/gt/gt.txt')
    gt = reader.get_annotations(classes=['car'], only_not_parked=True)

    roi = cv2.imread('data/AICity_data/train/S03/c010/roi.jpg', cv2.IMREAD_GRAYSCALE)

    cap = cv2.VideoCapture('data/AICity_data/train/S03/c010/vdo.avi')
    video_length = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    start_frame = int(video_length * model_frac)
    end_frame = int(video_length)

    for method in methods:
        backSub = sota_bg_subtractor(method)
        for _ in trange(start_frame, desc='modelling background'):
            ret, img = cap.read()
            backSub.apply(img)

        if save_path:
            writer = imageio.get_writer(os.path.join(save_path, f'task3_method_'+method+'.gif'), fps=10)

        y_pred = []
        y_true = []
        for frame in trange(start_frame, end_frame, desc='evaluating frames'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, img = cap.read()

            mask = backSub.apply(img)
            mask = mask & roi
            mask = postprocess(mask)

            detections = bounding_boxes(mask, min_height, max_height, min_width, max_width, frame)
            annotations = gt.get(frame, [])

            if debug >= 1 or save_path:
                img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                for det in detections:
                    cv2.rectangle(img, (det.xtl, det.ytl), (det.xbr, det.ybr), (0, 255, 0), 2)
                for det in annotations:
                    cv2.rectangle(img, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ybr)), (0, 0, 255), 2)

                if save_path:
                    writer.append_data(img)
                elif debug == 1:
                    cv2.imshow('result', img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            y_pred.append(detections)
            y_true.append(annotations)

        cv2.destroyAllWindows()
        if save_path:
            writer.close()

        ap, prec, rec = mean_average_precision(y_true, y_pred, classes=['car'])
        print(f'Method: {method}, AP: {ap:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}')


def task4(adaptive, random_search, model_frac=0.25, save_path=None, min_width=120, max_width=800, min_height=100, max_height=600, debug=0):
    """
    Color modelling
    """
    color_space = 'yuv'
    channels = [1, 2]
    n_ch = len(channels)

    # Read information
    reader = AICityChallengeAnnotationReader(path='data/AICity_data/train/S03/c010/gt/gt.txt')
    gt = reader.get_annotations(classes=['car'], only_not_parked=True)
    roi = cv2.imread('data/AICity_data/train/S03/c010/roi.jpg', cv2.IMREAD_GRAYSCALE)

    # Model Background
    bg_model = SingleGaussianBackgroundModel(video_path='data/AICity_data/train/S03/c010/vdo.avi',
                                             color_space=color_space, channels=channels, resize=None)
    video_length = bg_model.length
    bg_model.fit(start=0, length=int(video_length * 0.25))

    # Video length
    start_frame = int(video_length * model_frac)
    end_frame = video_length

    # hyperparameter search
    if random_search:
        alphas = np.random.choice(np.linspace(2, 4, 50), 25)
        rhos = np.random.choice(np.linspace(0.001, 0.1, 50), 25) if adaptive else [0]
        combinations = [(alpha, rho) for alpha, rho in zip(alphas, rhos)]
    else:
        alphas = [0.5, 1, 2]
        rhos = [0.005, 0.01] if adaptive else [0]
        combinations = [(alpha, rho) for alpha in alphas for rho in rhos]

    for alpha, rho in combinations:
        y_true = []
        y_pred = []

        if save_path:
            gif_name = f'task3_alpha_{str(alpha)}_rho_{str(rho)}_color_{color_space}_channels_{str(n_ch)}_{time.time()}.gif'
            writer = imageio.get_writer(os.path.join(save_path, gif_name), fps=25)

        for frame in trange(start_frame, end_frame, desc=f'obtaining foreground and detecting objects. Alpha {alpha} Rho {rho}'):

            frame_img, mask, _ = bg_model.evaluate(frame=frame, alpha=alpha)
            mask = mask & roi
            non_post_mask = mask
            mask = postprocess(mask)

            detections = bounding_boxes(mask, min_height, max_height, min_width, max_width, frame)
            annotations = gt.get(frame, [])

            if save_path:
                img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

                for det in detections:
                    cv2.rectangle(img, (det.xtl, det.ytl), (det.xbr, det.ybr), (0, 255, 0), 3)

                for det in annotations:
                    cv2.rectangle(img, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ybr )), (0, 0, 255), 2)

                writer.append_data(img)

                if debug >=1:
                    shape = (480, 270)
                    cv2.imshow(f'BGR Image', cv2.resize(img, shape))
                    cv2.imshow(f'Segmentation using {color_space}', cv2.resize(non_post_mask, shape))
                    cv2.imshow(f'Segmentation Morphed using {color_space}', cv2.resize(mask, shape))

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                y_pred.append(detections)
                y_true.append(annotations)

        cv2.destroyAllWindows()
        if save_path:
            writer.close()

        ap, prec, rec = mean_average_precision(y_true, y_pred, classes=['car'])
        print(f'alpha: {alpha:.1f}, rho: {rho:.3f}, AP: {ap:.4f}')


if __name__ == '__main__':
    #task1(debug=1)
    task2(debug=1)
    #task3(['MOG', 'MOG2', 'LSBP', 'GMG', 'KNN', 'GSOC', 'CNT'], debug=1)
    #task4(adaptive=True, random_search=False, save_path='results/week2', debug=1)
