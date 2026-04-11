import numpy as np
import numpy.typing as npt


def offset_times(times: npt.NDArray, offsets: npt.NDArray, labels: npt.NDArray):
    if len(labels) == 0:
        return times
    times2 = np.zeros_like(times)
    for k in range(1, int(np.max(labels)) + 1):
        inds = np.where(labels == k)[0]
        times2[inds] = times[inds] + offsets[k - 1]
    return times2
