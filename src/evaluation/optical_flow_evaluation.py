import numpy as np


def get_msen_pepn(flow_pred: np.ndarray, flow_gt: np.ndarray, th: int = 3) -> (np.ndarray, np.ndarray, float, float):
    """
    Mean Square Error in Non-occluded areas
    """

    # compute mse, filtering discarded vectors
    u_diff = flow_gt[:, :, 0] - flow_pred[:, :, 0]
    v_diff = flow_gt[:, :, 1] - flow_pred[:, :, 1]
    squared_error = np.sqrt(u_diff**2 + v_diff**2)

    # discard vectors which from occluded areas (occluded = 0)
    non_occluded_idx = flow_gt[:, :, 2] != 0
    err_non_occ = squared_error[non_occluded_idx]

    msen = np.mean(err_non_occ)
    pepn = get_pepn(err_non_occ, len(err_non_occ), th)

    return squared_error, err_non_occ, msen, pepn


def get_pepn(err: np.ndarray, n_pixels: int, th: int) -> float:
    """
    Percentage of Erroneous Pixels in Non-occluded areas
    """

    return (np.sum(err > th) / n_pixels) * 100
