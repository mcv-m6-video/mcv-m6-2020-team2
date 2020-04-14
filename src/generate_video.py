import os
import cv2
from threading import Thread

from utils.aicity_reader import AICityChallengeAnnotationReader


def run(cap, cam, start, end, dets):
    out = cv2.VideoWriter(f'{cam}_tot.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 10, (int(cap.get(3)),int(cap.get(4))))

    for frame in range(start, end):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)

        _, img = cap.read()
        detections_on_frame = dets.get(frame, [])
        for det in detections_on_frame:
            cv2.rectangle(img, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ybr)), (0, 255, 0), 2)
            cv2.putText(img, str(det.id), (int(det.xtl), int(det.ytl)), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)

        out.write(img)

    cap.release()
    out.release()


def main():
    path = "tracking/mtmc/results/S03"
    cams = ['c010', 'c011', 'c012', 'c013', 'c014', 'c015']

    cam1 = cams[1]
    cam2 = cams[2]
    cap1 = cv2.VideoCapture(f'../data/AIC20_track3/train/S03/{cam1}/vdo.avi')
    cap2 = cv2.VideoCapture(f'../data/AIC20_track3/train/S03/{cam2}/vdo.avi')

    reader = AICityChallengeAnnotationReader(path=os.path.join(path, cam1, "results.txt"))
    dets1 = reader.get_annotations(classes=['car'])
    reader = AICityChallengeAnnotationReader(path=os.path.join(path, cam2, "results.txt"))
    dets2 = reader.get_annotations(classes=['car'])

    vid1 = Thread(target=lambda: run(cap1, cam1, 211, 2221, dets1))
    vid2 = Thread(target=lambda: run(cap2, cam2, 211, 2221, dets2))

    vid1.start()
    vid2.start()

    vid1.join()
    vid2.join()

if __name__ == "__main__":
    main()
