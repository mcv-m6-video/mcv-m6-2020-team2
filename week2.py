import os

import numpy as np
import cv2
from tqdm import trange
import imageio

from src.utils.aicity_reader import AICityChallengeAnnotationReader
from src.segmentation.background_estimation import SingleGaussianBackgroundModel, sota_bg_subtractor
from src.utils.processing import postprocessing, bounding_boxes
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
            mask = postprocessing(mask)

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


def task3(methods, model_frac=0.25, min_width=120, max_width=800, min_height=100, max_height=600, history=10, debug=0):
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

        y_pred = []
        y_true = []
        for frame in trange(start_frame, end_frame, desc='evaluating frames'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, img = cap.read()

            mask = backSub.apply(img)
            mask = mask & roi
            mask = postprocessing(mask)

            detections = bounding_boxes(mask, min_height, max_height, min_width, max_width, frame)
            annotations = gt.get(frame, [])

            if debug >= 1:
                img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                for det in detections:
                    cv2.rectangle(img, (det.xtl, det.ytl), (det.xbr, det.ybr), (0, 255, 0), 2)
                for det in annotations:
                    cv2.rectangle(img, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ybr)), (0, 0, 255), 2)

                cv2.imshow('result', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            y_pred.append(detections)
            y_true.append(annotations)

        cv2.destroyAllWindows()

        ap, prec, rec = mean_average_precision(y_true, y_pred, classes=['car'])
        print(f'Method: {method}, AP: {ap:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}')


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
        frame_img, segmentation, _ = bg_model.evaluate(frame=frame, alpha=alpha)
        segmentation_postprocessed = postprocessing(segmentation)

        y_pred.append(bounding_boxes(segmentation, min_height=100, max_height=600, min_width=120, max_width=800, frame=frame))
        y_true.append(gt.get(frame, []))

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
        cv2.imshow(f'Segmentation Morphed using {color_space}', cv2.resize(segmentation_postprocessed, shape))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    ap, prec, rec = mean_average_precision(y_true, y_pred, classes=['car'])
    print(f'AP: {ap:.4f}')


if __name__ == '__main__':
    #task1(debug=1)
    #task2(debug=1)
    task3(['MOG', 'MOG2', 'LSBP', 'GMG', 'KNN', 'GSOC', 'CNT'], debug=1)
    #task4()
