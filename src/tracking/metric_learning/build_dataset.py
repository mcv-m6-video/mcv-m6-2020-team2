import glob
import os

import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.aicity_reader import AICityChallengeAnnotationReader, parse_annotations_from_txt, group_by_frame


def analyze_data(main_path, debug=False):
    areas = []
    heights = []
    widths = []
    for gt_file in glob.glob(os.path.join(main_path, "*", "*", "gt", "gt.txt")):
        video_path = gt_file.replace("gt\\gt.txt", "vdo.avi")
        cap = cv2.VideoCapture(video_path)
        reader = AICityChallengeAnnotationReader(path=gt_file)
        gt = reader.get_annotations(classes=['car'])
        for frame, detections in gt.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, img = cap.read()
            for det in detections:
                areas.append(det.area)
                heights.append(det.height)
                widths.append(det.width)
                if debug:
                    cv2.rectangle(img, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ybr)), (0, 255, 0), 2)
            if debug:
                cv2.imshow('result', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    plt.hist(areas, 100, facecolor='blue')
    plt.show()
    print("Min area", min(areas), " Min width", min(widths), " Min height", min(heights))


def downsample(img, max_height, max_width):
    height, width = img.shape[:2]
    # only shrink if img is bigger than required
    if max_height < height or max_width < width:
        # get scaling factor
        scaling_factor = max_height / float(height)
        if max_width / float(width) < scaling_factor:
            scaling_factor = max_width / float(width)
        # resize image
        img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return img


def generate_train_crops(root, save_path, train_seqs, val_seqs, width=128, height=128):

    def generate_crops(root, save_path):
        for cam in os.listdir(root):
            detections_by_frame = group_by_frame(parse_annotations_from_txt(os.path.join(root, cam, 'gt', 'gt.txt')))
            cap = cv2.VideoCapture(os.path.join(root, cam, 'vdo.avi'))
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            for _ in tqdm(range(length), desc=cam):
                frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                _, img = cap.read()
                if frame not in detections_by_frame:
                    continue

                for det in detections_by_frame[frame]:
                    if det.width >= width and det.height >= height:
                        id_path = os.path.join(save_path, str(det.id))
                        os.makedirs(id_path, exist_ok=True)

                        roi = img[int(det.ytl):int(det.ybr), int(det.xtl):int(det.xbr)]
                        resized = cv2.resize(roi, (width, height))
                        cv2.imwrite(os.path.join(id_path, f'{cam}_{frame}.png'), resized)

    for seq in train_seqs:
        generate_crops(root=os.path.join(root, seq), save_path=os.path.join(save_path, 'train'))
    for seq in val_seqs:
        generate_crops(root=os.path.join(root, seq), save_path=os.path.join(save_path, 'val'))


if __name__ == '__main__':
    # analyze_data('data/AIC20_track3/train', debug=True)
    generate_train_crops('../../../data/AIC20_track3/train', '../../../data/metric_learning',
                         train_seqs=['S01', 'S04'], val_seqs=['S03'])
