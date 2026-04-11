import torch
import numpy as np


def offset_times(times: torch.Tensor, offsets, labels):
    if len(labels) == 0:
        return times
    times2 = torch.zeros_like(times)
    offsets_tensor = torch.as_tensor(offsets, device=times.device)
    labels_tensor = torch.as_tensor(labels, device=times.device)
    for k in range(1, int(np.max(labels)) + 1):
        inds = torch.where(labels_tensor == k)[0]
        times2[inds] = times[inds] + offsets_tensor[k - 1]
    return times2
