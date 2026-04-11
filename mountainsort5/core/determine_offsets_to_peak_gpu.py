import torch
import numpy as np


def determine_offsets_to_peak(templates: torch.Tensor, *, detect_sign: int, T1: int):
    K = templates.shape[0]

    if detect_sign < 0:
        A = -templates
    elif detect_sign > 0:  # pragma: no cover
        A = templates  # pragma: no cover
    else:
        A = torch.abs(templates)  # pragma: no cover

    offsets_to_peak = np.zeros((K,), dtype=np.int32)
    for k in range(K):
        peak_channel = int(torch.argmax(torch.max(A[k], dim=0).values).item())
        peak_time = int(torch.argmax(A[k][:, peak_channel]).item())
        offsets_to_peak[k] = peak_time - T1
    return offsets_to_peak
