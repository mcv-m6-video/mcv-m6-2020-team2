import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from src.evaluation.intersection_over_union import mean_intersection_over_union


def video_iou_plot(gt, det, video_path, title='', save_path=None):
    frames = list(det.keys())
    overlaps = []

    for frame in frames:
        boxes1 = [d.bbox for d in gt.get(frame)]
        boxes2 = [d.bbox for d in det.get(frame, [])]
        iou = mean_intersection_over_union(boxes1, boxes2)
        overlaps.append(iou)

    cap = cv2.VideoCapture(video_path)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    fig, ax = plt.subplots(2, 1, figsize=(10, 5))
    image = ax[0].imshow(np.zeros((height, width)))
    line, = ax[1].plot(frames, overlaps)
    artists = [image, line]

    def update(i):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames[i])
        ret, img = cap.read()
        for d in gt[frames[i]]:
            cv2.rectangle(img, (int(d.xtl), int(d.ytl)), (int(d.xbr), int(d.ybr)), (0, 255, 0), 2)
        for d in det[frames[i]]:
            cv2.rectangle(img, (int(d.xtl), int(d.ytl)), (int(d.xbr), int(d.ybr)), (0, 0, 255), 2)
        artists[0].set_data(img[:, :, ::-1])
        artists[1].set_data(frames[:i + 1], overlaps[:i + 1])
        return artists

    ani = animation.FuncAnimation(fig, update, len(frames), interval=2, blit=True)

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_ylim(0, 1)
    ax[1].set_xlabel('#frame')
    ax[1].set_ylabel('mean IoU')
    fig.suptitle(title)
    if save_path is not None:
        ani.save(os.path.join(save_path, 'video_iou.gif'), writer='imagemagick')
    else:
        plt.show()


def optical_flow_magnitude_plot(flow_image: np.ndarray, frame_id: str, path: str = "", title="", dilate=False):
    if len(flow_image.shape) > 2:
        magnitude, angle = cv2.cartToPolar(flow_image[:, :, 0], flow_image[:, :, 1])
        flow_image = magnitude

    if dilate:
        kernel = np.ones((3, 3), np.uint8)
        flow_image = cv2.dilate(flow_image, kernel, iterations=1)

    plt.imshow(flow_image, cmap='pink')
    plt.axis('off')
    plt.title(f'{title}-{frame_id}')
    plt.savefig(os.path.join(path, f'{title.lower()}_{frame_id}.png')) if path else plt.show()
    plt.close()


def optical_flow_arrow_plot(gray_frame: np.ndarray, flow_image: np.ndarray, type: str, frame_id: str,
                            sampling_step: int, path: str):
    flow_image = cv2.resize(flow_image, (0, 0), fx=1. / sampling_step, fy=1. / sampling_step)
    u = flow_image[:, :, 0]
    v = flow_image[:, :, 1]

    width = np.arange(0, flow_image.shape[1] * sampling_step, sampling_step)
    height = np.arange(0, flow_image.shape[0] * sampling_step, sampling_step)
    x, y = np.meshgrid(width, height)
    max_vect_length = max(np.max(u), np.max(v))

    plt.figure()
    plt.quiver(x, y, u, -v, np.hypot(u, v), scale=max_vect_length * sampling_step, cmap='rainbow')
    plt.imshow(gray_frame, alpha=0.8, cmap='gray')
    plt.title(f'Flow Results {type}-{frame_id}')
    plt.axis('off')
    plt.savefig(os.path.join(path, f'flow_results_{type}-{frame_id}.png')) if path else plt.show()
    plt.close()


def histogram_with_mean_plot(title: str, idx: str, values: float, mean_value: float, save_path=None):
    plt.figure()
    plt.title(f'{title}-{idx}')
    plt.hist(values, 25, color="skyblue")
    plt.axvline(mean_value, color='g', linestyle='dashed', linewidth=1, label=f'MSEN {round(mean_value, 1)}')
    plt.legend()
    plt.savefig(os.path.join(save_path, f'histogram_{idx}.png')) if save_path else plt.show()
    plt.close()

def plot_3d(x, y, z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    l = ax.scatter(x, y, z, c=z, cmap= plt.cm.cool)
    ax.set_xlabel("alpha threshold")
    ax.set_ylabel("rho")
    ax.set_zlabel("mAP")
    fig.colorbar(l)
    plt.show()

if __name__ == '__main__':
    alpha=[5, 4.1, 5.9, 6.4, 6.2, 5.8, 3.5, 6.8, 5.3, 6.6, 2.8, 6.2, 5, 6.4, 5.2, 3, 5.4, 2, 5.3, 4, 2.5, 2.4, 4.8, 3.8, 4.2]
    rho = [0.076, 0.019, 0.049, 0.084, 0.017, 0.084, 0.033, 0.094, 0.072, 0.072, 0.088, 0.076, 0.072, 0.025, 0.013, 0.021, 0.037, 0.041, 0.011, 0.094, 0.029, 0.052, 0.043, 0.031, 0.037]
    mAP = [0.2466, 0.2587, 0.17, 0.1198, 0.1531, 0.167, 0.2059, 0.1043, 0.1795, 0.1146, 0.1902, 0.1318, 0.2057, 0.1481, 0.33, 0.5445, 0.2584, 0.0558, 0.2979, 0.4102, 0.1117, 0.1022, 0.2566, 0.3971, 0.361]
    plot_3d(alpha, rho, mAP)