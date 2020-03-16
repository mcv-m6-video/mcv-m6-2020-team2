import os
import numpy as np
import imageio
import cv2
from torchvision.models import detection
from torchvision.transforms import transforms
import torch
import time
from tqdm import trange


from src.evaluation.intersection_over_union import bb_intersecion_over_union
from src.evaluation.average_precision import mean_average_precision
from src.utils.aicity_reader import AICityChallengeAnnotationReader
from src.tracking.tracking import update_tracks_by_overlap
from src.utils.detection import Detection
from src.utils.plotutils import video_iou_plot
from src.tracking.sort import Sort


def task1_1(architecture, start=0, length=None, save_path='results/week3', gpu=0, visualize=False):
    ''' Object detection: Off-the-shelf '''
    tensor = transforms.ToTensor()

    if architecture.lower() == 'fasterrcnn':
        model = detection.fasterrcnn_resnet50_fpn(pretrained=True)

    elif architecture.lower() == 'maskrcnn':
        model = detection.maskrcnn_resnet50_fpn(pretrained=True)
    else:
        raise ValueError(architecture)
    save_path = os.path.join(save_path, architecture)

    # Read Video and prepare ground truth
    cap = cv2.VideoCapture('data/AICity_data/train/S03/c010/vdo.avi')
    if not length:
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    reader = AICityChallengeAnnotationReader(path='data/ai_challenge_s03_c010-full_annotation.xml')
    gt = reader.get_annotations(classes=['car'])
    gt = {frame: gt[frame] for frame in range(start, start + length)}

    # Start Inference
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    model.to(device)
    model.eval()
    detections = {}
    y_true, y_pred = [], []
    with torch.no_grad():
        for frame in range(start, length):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, img = cap.read()

            # Transform input to tensor
            print(f'Predict: {frame}')
            start_t = time.time()

            x = [tensor(img).to(device)]
            preds = model(x)[0]
            print(f"Inference time per frame: {round(time.time()-start_t, 2)}")

            # filter car predictions and confidences
            joint_preds = list(zip(preds["labels"], preds["boxes"], preds["scores"]))
            car_det = list(filter(lambda x: x[0] == 3, joint_preds))
            car_det = list(filter(lambda x: x[2] > 0.80, car_det))

            detections[frame] = []
            for det in car_det:
                detections[frame].append(Detection(frame=frame,
                                                    id=frame,
                                                    label='car',
                                                    xtl=float(det[1][0]),
                                                    ytl=float(det[1][1]),
                                                    xbr=float(det[1][2]),
                                                    ybr=float(det[1][3]),
                                                    score=det[2]))

            # Prepare for mean_avg_precision
            annotations = gt.get(frame, [])
            y_pred.append(detections[frame])
            y_true.append(annotations)

    ap, prec, rec = mean_average_precision(y_true, y_pred, classes=['car'])
    print(f'Network: {architecture}, AP: {ap:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}')

    if visualize:
        print(f'Saving result to {save_path}')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        video_iou_plot(gt, detections, video_path='data/AICity_data/train/S03/c010/vdo.avi', title=f'{architecture} detections',
                       save_path=save_path)

def task1_2():
    '''Object detection: Fine-tune to your data'''
    return

def task2_1(save_path=None, debug=0):
    '''Object tracking
     2.1. Tracking by overlap
    '''

    cap = cv2.VideoCapture('data/AICity_data/train/S03/c010/vdo.avi')
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    reader = AICityChallengeAnnotationReader(path='data/AICity_data/train/S03/c010/gt/gt.txt')
    gt = reader.get_annotations(classes=['car'], only_not_parked=True)
    reader = AICityChallengeAnnotationReader(path=f'data/AICity_data/train/S03/c010/det/det_yolo3.txt')
    annotations = reader.get_annotations(classes=['car'], only_not_parked=True)

    if save_path:
        writer = imageio.get_writer(os.path.join(save_path, f'task21.gif'), fps=10)

    y_true = []
    y_pred = []
    tracks = []
    max_track = 0
    for frame in trange(217, video_length, desc='evaluating frames'):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, img = cap.read()

        new_detections = annotations.get(frame, [])
        tracks, frame_tracks, max_track = update_tracks_by_overlap(tracks, new_detections, max_track)

        frame_detections=[]
        for track in frame_tracks:
            det = track.last_detection()
            frame_detections.append(det)
            if debug >= 1 or save_path:
                cv2.rectangle(img, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ybr)), track.color, 2)
                cv2.rectangle(img, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ytl) - 15), track.color, -2)
                cv2.putText(img, str(det.id), (int(det.xtl), int(det.ytl)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 2)

        y_pred.append(frame_detections)
        y_true.append(gt.get(frame, []))

        if save_path:
            writer.append_data(cv2.resize(img, (600, 350)))
        elif debug >= 1:
            cv2.imshow('result', cv2.resize(img, (900, 600)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    if save_path:
        writer.close()

    ap, prec, rec = mean_average_precision(y_true, y_pred, classes=['car'])
    print(f'AP: {ap:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}')

def task2_2(save_path=None, debug=0):
    '''Object tracking
     2.2. Tracking with a Kalman filter
    '''

    cap = cv2.VideoCapture('data/AICity_data/train/S03/c010/vdo.avi')
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    reader = AICityChallengeAnnotationReader(path='data/AICity_data/train/S03/c010/gt/gt.txt')
    gt = reader.get_annotations(classes=['car'], only_not_parked=True)
    reader = AICityChallengeAnnotationReader(path=f'data/AICity_data/train/S03/c010/det/det_yolo3.txt')
    annotations = reader.get_annotations(classes=['car'], only_not_parked=True)

    if save_path:
        writer = imageio.get_writer(os.path.join(save_path, f'task22.gif'), fps=10)

    kalman_tracker = Sort()
    y_true = []
    y_pred = []
    y_pred_kalman = []
    for frame in trange(217, video_length, desc='evaluating frames'):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, img = cap.read()

        detections_on_frame = annotations.get(frame, [])

        detections_formatted = np.array([[det.xtl, det.ytl, det.xbr, det.ybr, det.score] for det in detections_on_frame])
        frame_tracks = kalman_tracker.update(detections_formatted)

        frame_detections = []
        for track_det in frame_tracks:
            det = Detection(frame, int(track_det[4]), 'car', track_det[0], track_det[1], track_det[2], track_det[3], find_closest_bb(track_det[:4], detections_on_frame).score)
            frame_detections.append(det)
            if debug >= 1 or save_path:
                cv2.rectangle(img, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ybr)), (0, 255, 0), 2)
                cv2.rectangle(img, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ytl) - 15), (0, 255, 0), -2)
                cv2.putText(img, str(det.id), (int(det.xtl), int(det.ytl)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 2)

        y_pred_kalman.append(frame_detections)
        y_pred.append(detections_on_frame)
        y_true.append(gt.get(frame, []))

        if save_path:
            writer.append_data(cv2.resize(img, (600, 350)))
        elif debug >= 1:
            cv2.imshow('result', cv2.resize(img, (900, 600)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    if save_path:
        writer.close()

    ap, prec, rec = mean_average_precision(y_true, y_pred, classes=['car'])
    print(f'Original AP: {ap:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}')
    ap, prec, rec = mean_average_precision(y_true, y_pred_kalman, classes=['car'])
    print(f'After Kalman filter AP: {ap:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}')

def find_closest_bb(bb, bb_list):
    ''' Returns the bounding box in bb_list closest to bb '''
    max_iou = 0
    best_bb = bb_list[0]
    for b in bb_list:
        iou = bb_intersecion_over_union(bb, [b.xtl, b.ytl, b.xbr, b.ybr])
        # print('\t',bb,[b.xtl, b.ytl, b.xbr, b.ybr],iou)
        if iou > max_iou:
            max_iou = iou
            best_bb = b
    return best_bb

if __name__ == '__main__':
    #task1_1(model_name='mask', start=0, length=1)
    task2_2(debug=0)
