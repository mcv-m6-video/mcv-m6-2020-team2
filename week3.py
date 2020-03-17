import os
import time
from collections import defaultdict

import numpy as np
import cv2
import imageio
from tqdm import trange

import torch
from torchvision.models import detection
from torchvision.transforms import transforms

from src.detection.finetuning import get_data_loaders, get_model, train, evaluate
from src.tracking.tracking import update_tracks_by_overlap
from src.tracking.sort import Sort
from src.evaluation.average_precision import mean_average_precision
from src.evaluation.idf1 import MOTAcumulator
from src.utils.aicity_reader import AICityChallengeAnnotationReader
from src.utils.detection import Detection
from src.utils.plotutils import video_iou_plot
from src.utils.non_maximum_supression import get_nms

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def task1_1(architecture, start=0, length=None, save_path='results/week3', gpu=0, visualize=False, save_detection='detection_results/'):
    """
    Object detection: off-the-shelf
    """

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

    if save_detection:
        path = os.path.join(save_detection, architecture)
        if not os.path.exists(path):
            os.makedirs(path)
        detection_file  = open(f'{path}/{architecture.lower()}.txt', 'w')

    with torch.no_grad():
        for frame in range(start, length):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, img = cap.read()

            # Transform input to tensor
            print(f'Predict: {frame}')
            start_t = time.time()

            x = [tensor(img).to(device)]
            preds = model(x)[0]
            print(f'Inference time per frame: {round(time.time() - start_t, 2)}')

            # filter car predictions and confidences
            joint_preds = list(zip(preds['labels'], preds['boxes'], preds['scores']))
            car_det = list(filter(lambda x: x[0] == 3, joint_preds))
            # car_det = list(filter(lambda x: x[2] > 0.70, car_det))
            car_det = get_nms(car_det, 0.7)

            # add detections
            detections[frame] = []
            for det in car_det:
                det_obj = Detection(frame=frame,
                                   id=None,
                                   label='car',
                                   xtl=float(det[1][0]),
                                   ytl=float(det[1][1]),
                                   xbr=float(det[1][2]),
                                   ybr=float(det[1][3]),
                                   score=det[2])

                detections[frame].append(det_obj)

                cv2.imshow('image', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                if save_detection:
                    detection_file.write(f"{frame},-1,{det_obj.xtl},{det_obj.ytl},{det_obj.width},{det_obj.height},{det_obj.score},-1,-1,-1\n")

            y_pred.append(detections[frame])
            y_true.append(gt.get(frame, []))

    ap, prec, rec = mean_average_precision(y_true, y_pred, classes=['car'])
    print(f'Network: {architecture}, AP: {ap:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}')

    if visualize:
        print(f'Saving result to {save_path}')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        video_iou_plot(gt, detections, video_path='data/AICity_data/train/S03/c010/vdo.avi',
                       title=f'{architecture} detections',
                       save_path=save_path)

    cv2.destroyAllWindows()

    if save_detection:
        detection_file.close()


def task1_2(finetune=True, architecture='maskrcnn', save_path=None):
    """
    Object detection: fine-tuning
    """

    np.random.seed(42)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train_loader, test_loader = get_data_loaders(root='data')
    model = get_model(architecture=architecture, finetune=finetune, num_classes=len(train_loader.dataset.classes))
    model.to(device)

    if finetune:
        train(model, train_loader, test_loader, device, save_path=save_path)
    else:
        evaluate(model, test_loader, device, save_path=save_path)


def task2_1(save_path=None, debug=0):
    """
    Object tracking: tracking by overlap
    """

    cap = cv2.VideoCapture('data/AICity_data/train/S03/c010/vdo.avi')
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    reader = AICityChallengeAnnotationReader(path='data/ai_challenge_s03_c010-full_annotation.xml')
    gt = reader.get_annotations(classes=['car'], only_not_parked=False)
    reader = AICityChallengeAnnotationReader(path=f'data/AICity_data/train/S03/c010/det/det_yolo3.txt')
    annotations = reader.get_annotations(classes=['car'], only_not_parked=False)

    if save_path:
        writer = imageio.get_writer(os.path.join(save_path, f'task21.gif'), fps=10)

    accumulator = MOTAcumulator()
    y_true = []
    y_pred = []
    y_pred_refined = []
    tracks = []
    max_track = 0
    for frame in trange(217, video_length, desc='evaluating frames'):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, img = cap.read()

        detections_on_frame = annotations.get(frame, [])
        tracks, frame_tracks, max_track = update_tracks_by_overlap(tracks, detections_on_frame, max_track)

        frame_detections = []
        for track in frame_tracks:
            det = track.last_detection()
            frame_detections.append(det)
            if debug >= 1 or save_path:
                cv2.rectangle(img, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ybr)), track.color, 2)
                cv2.rectangle(img, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ytl) - 15), track.color, -2)
                cv2.putText(img, str(det.id), (int(det.xtl), int(det.ytl)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                for dd in track.track:
                    cv2.circle(img, dd.center, 5, track.color, -1)

        y_pred_refined.append(frame_detections)
        y_pred.append(detections_on_frame)
        y_true.append(gt.get(frame, []))

        accumulator.update(y_true[-1], y_pred_refined[-1])

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
    ap, prec, rec = mean_average_precision(y_true, y_pred_refined, classes=['car'])
    print(f'After refinement AP: {ap:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}')
    print('\nAdditional metrics:')
    print(accumulator.get_idf1())


def task2_2(debug=False):
    """
    Object tracking: tracking with a Kalman filter
    """

    reader = AICityChallengeAnnotationReader(path='data/ai_challenge_s03_c010-full_annotation.xml')
    gt = reader.get_annotations(classes=['car'])
    reader = AICityChallengeAnnotationReader(path=f'results/week3/det_maskrcnn_finetuning.txt')
    dets = reader.get_annotations(classes=['car'])

    cap = cv2.VideoCapture('data/AICity_data/train/S03/c010/vdo.avi')

    tracker = Sort()
    tracks = defaultdict(list)

    y_true = []
    y_pred = []
    acc = MOTAcumulator()
    for frame in dets.keys():

        detections = dets.get(frame, [])

        new_detections = tracker.update(np.array([[*d.bbox, d.score] for d in detections]))
        new_detections = [Detection(frame, int(d[-1]), 'car', *d[:4]) for d in new_detections]

        y_true.append(gt.get(frame, []))
        y_pred.append(new_detections)

        acc.update(y_true[-1], y_pred[-1])

        if debug:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, img = cap.read()
            for d in new_detections:
                tracks[d.id].append(d.bbox)
                np.random.seed(d.id)
                color = tuple(np.random.randint(0, 256, 3).tolist())
                for dd in tracks[d.id]:
                    cv2.circle(img, (int((dd[0]+dd[2])/2), int((dd[1]+dd[3])/2)), 3, color, -1)

            cv2.imshow('image', cv2.resize(img, (900, 600)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    ap, prec, rec = mean_average_precision(y_true, y_pred, classes=['car'])
    idf1 = acc.get_idf1()
    print(f"AP: {ap:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, IDF1: {idf1:.4f}")


if __name__ == '__main__':
    # task1_1(architecture='maskrcnn', start=0, length=2)
    # task1_2(finetune=True, architecture='maskrcnn', save_path='results/week3/det_maskrcnn_finetuning.txt')

    #task2_1(save_path='results/week3/', debug=0)
    task2_2(debug=True)
