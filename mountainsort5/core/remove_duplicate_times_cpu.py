import numpy as np
import numpy.typing as npt


def remove_duplicate_times(times: npt.NDArray, labels: npt.NDArray):
    if len(times) == 0:
        return times, labels
    inds = np.where(np.diff(times) > 0)[0]
    inds = np.concatenate([np.array([0]), inds + 1])
    return times[inds], labels[inds]
