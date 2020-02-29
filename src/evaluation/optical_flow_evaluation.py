import numpy as np
from matplotlib import pyplot as plt


def get_msen_pepn(flow_pred:np.ndarray, flow_gt:np.ndarray, frame_id:str='', th:int=3, plot:bool=False) -> (float, float):
    """
    Mean Square Error in Non-occluded areas
    """
    non_occluded_idx = flow_gt[:, :, 2] != 0 # Discard vectors which from occluded areas (occluded = 0)

    # Compute mse, filtering discarded vectors
    u_diff =  (flow_gt[:, :, 0] - flow_pred[:,:,0])**2
    v_diff =  (flow_gt[:, :, 1] - flow_pred[:,:,1])**2
    squared_error = np.sqrt(u_diff + v_diff)

    err_non_occ = squared_error[non_occluded_idx]

    msen = np.mean(err_non_occ)
    pepn = get_pepn(err_non_occ, len(err_non_occ), th)

    if plot:
        _plot_hist(frame_id, err_non_occ, msen)
        _plot_err_image(frame_id, squared_error)

    return msen, pepn

# Percentage of Erroneous Pixels in Non-occluded areas
def get_pepn(err: np.ndarray, n_pixels: int, th:int) -> float:
    return (np.sum(err > th) / n_pixels)*100


def _plot_hist(idx: str, err: float, msen:float):
    plt.figure()
    plt.title(f'Error Histogram {idx}')
    plt.hist(err, 25, color="skyblue")
    plt.axvline(msen, color='g', linestyle='dashed', linewidth=1, label=f'MSEN {round(msen, 1)}')
    plt.legend()
    plt.show()

def _plot_err_image(idx: str, err: float,):
    plt.figure()
    plt.title(f'Error image {idx}')
    plt.imshow(err)
    plt.colorbar()
    plt.tick_params(axis='both', labelbottom=False, labelleft=False)
    plt.show()