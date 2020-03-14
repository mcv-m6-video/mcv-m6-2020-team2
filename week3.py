import os
import imageio
import cv2
from torchvision.models import detection
from torchvision.transforms import transforms
import torch
import time

from src.evaluation.average_precision import mean_average_precision
from src.utils.aicity_reader import AICityChallengeAnnotationReader
from src.segmentation.tracking import tracking_by_overlap
from src.utils.detection import Detection
from src.utils.plotutils import video_iou_plot


def task1_1(model_name, start=0, length=None, save_path='results/week3', device=0):
    ''' Object detection: Off-the-shelf '''

    tensor = transforms.ToTensor()

    if model_name.lower() == 'fast':
        model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
        model_name = 'FastRCNN'

    elif model_name.lower() == 'mask':
        model = detection.maskrcnn_resnet50_fpn(pretrained=True)
        model_name = 'MaskRCNN'
    else:
        raise ValueError(model_name)
    save_path = os.path.join(save_path, model_name)

    # Read Video and prepare ground truth
    cap = cv2.VideoCapture('data/AICity_data/train/S03/c010/vdo.avi')
    if not length:
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    reader = AICityChallengeAnnotationReader(path='data/ai_challenge_s03_c010-full_annotation.xml')
    gt = reader.get_annotations(classes=['car'])
    gt = {frame: gt[frame] for frame in range(start, start + length)}

    # Start Inference
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        model = model.to('cuda:0')

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

            x = tensor(img).to('cuda:0') if torch.cuda.is_available() else tensor(img)
            preds = model([x])[0]
            print(f"Inference time per frame: {round(time.time()-start_t, 2)}")

            # filter car predictions and confidences
            joint_preds = list(zip(preds["labels"], preds["boxes"], preds["scores"]))
            car_det = list(filter(lambda x: x[0] == 3, joint_preds))
            car_det = list(filter(lambda x: x[2] > 0.70, car_det))

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
    print(f'Network: {model_name}, AP: {ap:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}')

    print(f'Saving result to {save_path}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    video_iou_plot(gt, detections, video_path='data/AICity_data/train/S03/c010/vdo.avi', title=f'{model_name} detections',
                   save_path=save_path)




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
    '''Object tracking:Tracking with a Kalman filter'''
    return

if __name__ == '__main__':
    task1_1(model_name='mask', start=0, length=1)
    # task2_1(debug=1)