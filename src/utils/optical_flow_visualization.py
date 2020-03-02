import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

def plot_optical_flow(gray_frame, flow_image, typee, frame_id, sampling_step, path):

    flow_image = cv2.resize(flow_image, (0, 0), fx=1./sampling_step, fy=1./sampling_step)
    u = flow_image[:, :, 0]
    v = flow_image[:, :, 1]

    width = np.arange(0, flow_image.shape[1]*sampling_step, sampling_step)
    height = np.arange(0, flow_image.shape[0]*sampling_step, sampling_step)
    x,y = np.meshgrid(width, height)

    max_vect_length = max(np.max(u), np.max(v))


    plt.figure()
    plt.quiver(x, y, u, -v, np.hypot(u,v), scale=max_vect_length*sampling_step , cmap='rainbow')
    plt.imshow(gray_frame, alpha=0.8, cmap='gray')
    plt.title(f'Flow Results {typee}-{frame_id}')
    plt.axis('off')

    plt.savefig(os.path.join(path, f'flow_results_{typee}-{frame_id}.png')) if path else plt.show()