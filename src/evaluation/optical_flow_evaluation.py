import numpy as np


def get_msen_pepn(flow_pred:np.ndarray, flow_gt:np.ndarray, th:int=3,) -> (np.ndarray, np.ndarray, float, float):
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

    return squared_error, err_non_occ, msen, pepn

# Percentage of Erroneous Pixels in Non-occluded areas
def get_pepn(err: np.ndarray, n_pixels: int, th:int) -> float:
    return (np.sum(err > th) / n_pixels)*100


