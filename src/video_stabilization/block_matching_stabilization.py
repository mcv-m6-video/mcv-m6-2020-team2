import numpy as np
import cv2
import imageio
from tqdm import trange

from src.optical_flow.block_matching_flow import block_matching_flow


def apply_camera_motion(frame, optical_flow, w, h, acc_t, method='average'):

    if method == 'average':
        average_optical_flow = np.array(optical_flow.mean(axis=0).mean(axis=0), dtype=int)
        acc_t += average_optical_flow
        H = np.float32([[1, 0, acc_t[0]], [0, 1, acc_t[1]]])
        frame_stabilized = cv2.warpAffine(frame, H, (w, h))   

    else:
        raise ValueError('Camera motion method not valid.')
    
    return frame_stabilized, acc_t

def block_matching_stabilization(cap, output_file, to_video, video_percentage=1):
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    w = 400
    h = 250
    method = 'average'

    if to_video:
        out = cv2.VideoWriter(f'{output_file}/block_matching_stabilization_'+method+'.mp4', cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))
    else:
        out = imageio.get_writer(f'{output_file}/block_matching_stabilization_'+method+'.gif', fps=fps)

    start = 0
    end = int(n_frames * video_percentage)
    previous_frame = None
    acc_t = np.zeros(2)
    for i in trange(start, end, desc="Apply stabilization and saving video"):
        success, frame = cap.read()
        frame = cv2.resize(frame, (w, h))
        if not success:
            break

        if i == 0:
            frame_stabilized = frame
        else:
            # Compute optical flow between frames
            optical_flow = block_matching_flow(previous_frame, frame)

            # Stabilize image
            frame_stabilized, acc_t = apply_camera_motion(frame, optical_flow, w, h, acc_t, method=method)

        previous_frame = frame

        if to_video:
            out.write(frame_stabilized)
        else:
            out.append_data(cv2.cvtColor(frame_stabilized, cv2.COLOR_BGR2RGB))

    if not to_video:
        out.close()


if __name__ == '__main__':
    img = cv2.imread(r'C:\Users\PC\Pictures\18_3.jpg')
    img2 = apply_camera_motion(img, [5.05, 8.54], img.shape[1], img.shape[0])
    cv2.imwrite(r'C:\Users\PC\Pictures\18_3_v2.png', img2)
