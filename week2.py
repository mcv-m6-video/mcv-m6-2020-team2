import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import trange

from src.utils.aicity_reader import AICityChallengeAnnotationReader
from src.evaluation.average_precision import mean_average_precision
from src.segmentation.background_estimation import GaussianModelling
from src.utils.processing import denoise, fill_holes, bounding_boxes


VIDEO_LENGTH = 2141

def task1(save_path=None, visualize=False):
    # EVALUATION CONSIDERING PARKED CARS

    reader = AICityChallengeAnnotationReader(path='data/ai_challenge_s03_c010-full_annotation.xml')
    gt = reader.get_annotations(classes=['car'])

    bg_model = GaussianModelling(video_path='data/AICity_data/train/S03/c010/vdo.avi')
    bg_model.fit(start=0, length=int(VIDEO_LENGTH*0.25))

    y_true = []
    y_pred = []
    for frame in trange(int(VIDEO_LENGTH*0.25), VIDEO_LENGTH, desc='obtaining foreground and detecting objects'):
        segmentation = bg_model.evaluate(frame=frame, alpha=4)
        segmentation_denoised = denoise(segmentation)
        segmentation_filled = fill_holes(segmentation_denoised)

        y_pred.append(bounding_boxes(segmentation_filled, frame=frame, min_area=200))
        y_true.append(gt[frame])

    ap = mean_average_precision(y_true, y_pred, classes=['car'])
    print(f'AP (considering parked cars): {ap:.4f}')

    if visualize:
        F = 600

        fig,ax = plt.subplots()
        ax.imshow(segmentation_filled)

        for item in y_pred[F-int(VIDEO_LENGTH*0.25)]:
            rect = patches.Rectangle((item.xtl,item.ytl),item.xbr-item.xtl,item.ybr-item.ytl,fill=False,linewidth=1,edgecolor='r')
            ax.add_patch(rect)
        for item in y_true[F-int(VIDEO_LENGTH*0.25)]:
            rect = patches.Rectangle((item.xtl,item.ytl),item.xbr-item.xtl,item.ybr-item.ytl,fill=False,linewidth=1,edgecolor='g')
            ax.add_patch(rect)

        ax.set_axis_off()
        plt.tight_layout()
        plt.show()

    # TODO: EVALUATION NOT CONSIDERING PARKED CARS

    return

def task2():
    #TODO Adaptive modeling
    return

def task3():
    '''
    Comparison with the state of the art
    '''

    method='MOG2'
    history=10

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
        raise ValueError(f"Unknown background estimation method {method}. Options are [MOG2, LSBP, GMG, KNN, GSOC, CNT]")

    cap = cv2.VideoCapture('data/AICity_data/train/S03/c010/vdo.avi')
    while cap.isOpened():
        retVal, frame = cap.read()

        fgmask = backSub.apply(frame, learningRate=1.0 / history)

        cv2.imshow('Foreground', cv2.resize(fgmask, (960, 540)))
        cv2.imshow('Original', cv2.resize(frame, (960, 540)))
        if cv2.waitKey(1) & 0xFF == 27:
            break

def task4():
    #TODO Color sequences
    return

if __name__ == '__main__':
    task3()