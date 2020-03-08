import cv2
from tqdm import trange
import matplotlib.pyplot as plt

from src.utils.aicity_reader import AICityChallengeAnnotationReader
from src.segmentation.background_estimation import SingleGaussianBackgroundModel
from src.utils.detection import Detection
from src.evaluation.average_precision import mean_average_precision
from src.utils.processing import denoise, fill_holes, bounding_boxes


def task1(model_frac=0.25, min_area=500, debug=0):
    reader = AICityChallengeAnnotationReader(path='data/ai_challenge_s03_c010-full_annotation.xml')
    gt = reader.get_annotations(classes=['car'], only_not_parked=True)

    bg_model = SingleGaussianBackgroundModel(video_path='data/AICity_data/train/S03/c010/vdo.avi')
    video_length = bg_model.length
    bg_model.fit(start=0, length=int(video_length * model_frac))

    roi = cv2.imread('data/AICity_data/train/S03/c010/roi.jpg', cv2.IMREAD_GRAYSCALE)

    start_frame = int(video_length * model_frac)
    end_frame = video_length
    y_true = []
    y_pred = []
    for frame in trange(start_frame, end_frame, desc='evaluating frames'):
        _, mask = bg_model.evaluate(frame=frame, rho=0)
        mask = mask & roi
        if debug >= 2:
            plt.imshow(mask); plt.show()

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        if debug >= 2:
            plt.imshow(mask); plt.show()

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for c in contours:
            if cv2.contourArea(c) < min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            detections.append(Detection(frame, None, 'car', x, y, x + w, y + h))
        annotations = gt.get(frame, [])

        if debug >= 1:
            img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            for det in detections:
                cv2.rectangle(img, (det.xtl, det.ytl), (det.xbr, det.ybr), (0, 255, 0), 2)
            for det in annotations:
                cv2.rectangle(img, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ybr)), (0, 0, 255), 2)
            cv2.imshow('frame', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        y_pred.append(detections)
        y_true.append(annotations)

    ap = mean_average_precision(y_true, y_pred, classes=['car'])
    print(f'AP: {ap:.4f}')


def task2():
    # TODO: Adaptive modeling
    return


def task3():
    """
    Comparison with the state of the art
    """

    method = 'MOG2'
    history = 10

    if method == 'MOG2':
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
        raise ValueError(
            f"Unknown background estimation method {method}. Options are [MOG2, LSBP, GMG, KNN, GSOC, CNT]")

    cap = cv2.VideoCapture('data/AICity_data/train/S03/c010/vdo.avi')
    while cap.isOpened():
        retVal, frame = cap.read()

        fgmask = backSub.apply(frame, learningRate=1.0 / history)

        cv2.imshow('Foreground', cv2.resize(fgmask, (960, 540)))
        cv2.imshow('Original', cv2.resize(frame, (960, 540)))
        if cv2.waitKey(1) & 0xFF == 27:
            break


def task4():
    shape = (480, 270)
    color_space = 'hsv'

    # Select which channels to use
    # all --> lambda img: img
    # only two --> lambda img: img[:,:,1:3]
    reshape_channels = lambda img: img[:, :, 0:2]
    alpha = 2

    reader = AICityChallengeAnnotationReader(path='data/ai_challenge_s03_c010-full_annotation.xml')
    gt = reader.get_annotations(classes=['car'], only_not_parked=True)

    bg_model = SingleGaussianBackgroundModel(video_path='data/AICity_data/train/S03/c010/vdo.avi', color_space=color_space)
    video_length = bg_model.length
    bg_model.fit(start=0, length=int(video_length * 0.25))

    y_true = []
    y_pred = []
    for frame in trange(int(video_length * 0.25), video_length, desc='obtaining foreground and detecting objects'):
        segmentation, frame_img = bg_model.evaluate(frame=frame, alpha=alpha)
        segmentation_denoised = denoise(segmentation)
        segmentation_filled = fill_holes(segmentation_denoised)

        y_pred.append(
            bounding_boxes(segmentation, frame=frame, min_height=100, max_height=600, min_width=120, max_width=800))
        if frame in gt.keys():
            y_true.append(gt[frame])
        else:
            y_true.append([])

        for i in range(frame_img.shape[-1]):
            cv2.imshow(f'Frame_{i}', cv2.resize(frame_img[:, :, i], shape))
        cv2.imshow('Segmentation', cv2.resize(segmentation_filled, shape))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    ap = mean_average_precision(y_true, y_pred, classes=['car'])
    print(f'AP: {ap:.4f}')


if __name__ == '__main__':
    task1(debug=1)
    #task3()
    #task4()
