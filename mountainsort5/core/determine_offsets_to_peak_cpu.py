import numpy as np
import numpy.typing as npt


def determine_offsets_to_peak(templates: npt.NDArray[np.float32], *, detect_sign: int, T1: int):
    K = templates.shape[0]

    if detect_sign < 0:
        A = -templates
    elif detect_sign > 0:  # pragma: no cover
        A = templates  # pragma: no cover
    else:
        A = np.abs(templates)  # pragma: no cover

    offsets_to_peak = np.zeros((K,), dtype=np.int32)
    for k in range(K):
        peak_channel = np.argmax(np.max(A[k], axis=0))
        peak_time = np.argmax(A[k][:, peak_channel])
        offsets_to_peak[k] = peak_time - T1
    return offsets_to_peak
