import os
import imageio
import cv2

from src.utils.aicity_reader import AICityChallengeAnnotationReader
from src.segmentation.tracking import tracking_by_overlap


def task1_1():
    ''' Object detection: Off-the-shelf '''
    return

def task1_2():
    '''Object detection: Fine-tune to your data'''
    return

def task2_1(save_path=None, debug=0):
    '''Object tracking: Tracking by overlap'''

    cap = cv2.VideoCapture('data/AICity_data/train/S03/c010/vdo.avi')
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    reader = AICityChallengeAnnotationReader(path=f'data/AICity_data/train/S03/c010/det/det_yolo3.txt')
    annotations = reader.get_annotations(classes=['car'], only_not_parked=True)

    if save_path:
        writer = imageio.get_writer(os.path.join(save_path, f'task21.gif'), fps=10)

    for frame in range(217, 250):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, img = cap.read()
        detections = tracking_by_overlap(annotations, frame)

        if debug >= 1 or save_path:
            for det in detections:
                cv2.rectangle(img, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ybr)), (0, 255, 0), 2)
                cv2.rectangle(img, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ytl)-15), (0, 255, 0), -2)
                cv2.putText(img, str(det.id), (int(det.xtl), int(det.ytl)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 2)

            if save_path:
                writer.append_data(cv2.resize(img,(600,350)))
            elif debug >= 1:
                cv2.imshow('result', cv2.resize(img,(900,600)))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cv2.destroyAllWindows()
    if save_path:
        writer.close()

def task2_2():
    '''Object tracking:Tracking with a Siamese network'''
    return

if __name__ == '__main__':
    task2_1(debug=1)