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

def task1(save_path=None):
    reader = AICityChallengeAnnotationReader(path='data/ai_challenge_s03_c010-full_annotation.xml')
    gt = reader.get_annotations(classes=['car'], only_not_parked=True)

    bg_model = GaussianModelling(video_path='data/AICity_data/train/S03/c010/vdo.avi')
    bg_model.fit(start=0, length=int(VIDEO_LENGTH*0.25))

    aps = []
    alphas = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    for alpha in alphas:
        print('\nalpha =', alpha)
        y_true = []
        y_pred = []
        for frame in trange(int(VIDEO_LENGTH*0.25), VIDEO_LENGTH, desc='obtaining foreground and detecting objects'):
            segmentation, _ = bg_model.evaluate(frame=frame, alpha=alpha)
            segmentation = fill_holes(segmentation)
            segmentation = denoise(segmentation)
            to_show = segmentation.copy()

            y_pred.append(bounding_boxes(segmentation, frame=frame, min_height=100, max_height=600, min_width=120, max_width=800))
            if frame in gt.keys():
                y_true.append(gt[frame])
            else:
                y_true.append([])

            ## PLOTS FOR TESTING
            # fig,ax = plt.subplots()
            # ax.imshow(to_show)

            # for item in y_pred[-1]:
            #     rect = patches.Rectangle((item.xtl,item.ytl),item.xbr-item.xtl,item.ybr-item.ytl,fill=False,linewidth=1,edgecolor='r')
            #     ax.add_patch(rect)
            # for item in y_true[-1]:
            #     rect = patches.Rectangle((item.xtl,item.ytl),item.xbr-item.xtl,item.ybr-item.ytl,fill=False,linewidth=1,edgecolor='g')
            #     ax.add_patch(rect)

            # ax.set_axis_off()
            # plt.tight_layout()
            # plt.show()
            # plt.waitforbuttonpress()
            # plt.cla()
            # plt.close()

        ap = mean_average_precision(y_true, y_pred, classes=['car'])
        print(f'AP: {ap:.4f}')

        aps.append(ap)

    plt.plot(alphas, aps)
    plt.xticks(alphas)
    plt.xlabel('alpha')
    plt.ylabel('AP')
    plt.show()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'ap_alpha.png'))

    return

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

    bg_model = GaussianModelling(video_path='data/AICity_data/train/S03/c010/vdo.avi', color_space=color_space, reshape_channels=reshape_channels)
    bg_model.fit(start=0, length=int(VIDEO_LENGTH * 0.25))

    y_true = []
    y_pred = []
    for frame in trange(int(VIDEO_LENGTH * 0.25), VIDEO_LENGTH, desc='obtaining foreground and detecting objects'):
        segmentation, frame_img = bg_model.evaluate(frame=frame, alpha=alpha)
        segmentation_denoised = denoise(segmentation)
        segmentation_filled = fill_holes(segmentation_denoised)

        y_pred.append(bounding_boxes(segmentation, frame=frame, min_height=100, max_height=600, min_width=120, max_width=800))
        if frame in gt.keys():
            y_true.append(gt[frame])
        else:
            y_true.append([])

        for i in range(frame_img.shape[-1]):
            cv2.imshow(f'Frame_{i}', cv2.resize(frame_img[:,:,i], shape))
        cv2.imshow('Segmentation', cv2.resize(segmentation_filled, shape))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    ap = mean_average_precision(y_true, y_pred, classes=['car'])
    print(f'AP: {ap:.4f}')




if __name__ == '__main__':
    #task1()
    #task3()
    task4()

