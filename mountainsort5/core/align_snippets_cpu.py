import numpy as np
import numpy.typing as npt

def align_snippets(snippets: npt.NDArray[np.float32], offsets: npt.NDArray, labels: npt.NDArray):
    if len(labels) == 0:
        return snippets
    snippets2 = np.zeros_like(snippets)
    for k in range(1, int(np.max(labels)) + 1):
        inds = np.where(labels == k)[0]
        snippets2[inds] = np.roll(snippets[inds], shift=offsets[k - 1], axis=1)
    return snippets2