import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

def arrow_optical_flow_plot(gray_frame:np.ndarray, flow_image:np.ndarray, typee:str, frame_id:str, sampling_step:int, path:str):

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
    plt.close()

def optical_flow_magnitud_plot(flow_image: np.ndarray, frame_id:str, path:str= "", title="", dilate=False):

    if len(flow_image.shape) > 2:
        magnitude, angle = cv2.cartToPolar(flow_image[:,:, 0], flow_image[:,:, 1])
        flow_image = magnitude

    if dilate:
        kernel = np.ones((3, 3), np.uint8)
        flow_image = cv2.dilate(flow_image, kernel, iterations=1)

    plt.imshow(flow_image, cmap='pink')
    plt.axis('off')
    plt.title(f'{title}-{frame_id}')
    plt.savefig(os.path.join(path, f'{title.lower()}_{frame_id}.png')) if path else plt.show()
    plt.close()


def histogram_with_mean_plot(title:str, idx:str, values: float, mean_value:float, save_path=None):
    plt.figure()
    plt.title(f'{title}-{idx}')
    plt.hist(values, 25, color="skyblue")
    plt.axvline(mean_value, color='g', linestyle='dashed', linewidth=1, label=f'MSEN {round(mean_value, 1)}')
    plt.legend()
    plt.savefig(os.path.join(save_path, f'histogram_{idx}.png')) if save_path else plt.show()
    plt.close()
