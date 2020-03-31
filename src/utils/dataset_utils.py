import os, glob
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.utils.aicity_reader import AICityChallengeAnnotationReader


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
    print("Min area",min(areas), " Min width", min(widths), " Min height", min(heights))

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

def generate_train_crops(main_path, max_height=100.0, max_width=100.0):
    for gt_file in glob.glob(os.path.join(main_path, "*", "*", "gt", "gt.txt")):
        print(gt_file)
        camera = gt_file.split("\\")[2]
        video_path = gt_file.replace("gt\\gt.txt", "vdo.avi")
        img_path = gt_file.replace(camera+"\\gt\\gt.txt", "img")
        if not os.path.exists(img_path):
            os.mkdir(img_path)
        cap = cv2.VideoCapture(video_path)
        # Read annotations
        reader = AICityChallengeAnnotationReader(path=gt_file)
        gt = reader.get_annotations(classes=['car'])
        for frame, detections in tqdm(gt.items(),position=0, leave=True):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, img = cap.read()
            for det in detections:
                path_to_save_det = img_path+"\\"+str(det.id)
                if not os.path.exists(path_to_save_det):
                    os.mkdir(path_to_save_det)
                # crop and resize car
                roi = img[int(det.ytl):int(det.ybr), int(det.xtl):int(det.xbr)]
                resized = downsample(roi, max_height, max_width)
                cv2.imwrite(path_to_save_det+"\\"+str(camera)+"_"+str(frame)+".png", resized)

if __name__ == '__main__':
    main_path = "data/AIC20_track3/train"
    #analyze_data(main_path, debug=True)
    generate_train_crops(main_path, max_height=100.0, max_width=100.0)
