import torch
import numpy as np

def align_snippets(snippets: torch.Tensor, offsets, labels):
    if len(labels) == 0:
        return snippets
    snippets2 = torch.zeros_like(snippets)
    max_label = int(np.max(labels))
    for k in range(1, max_label + 1):
        inds = np.where(labels == k)[0]
        if len(inds) > 0:
            snippets2[inds] = torch.roll(snippets[inds], shifts=int(offsets[k - 1]), dims=1)
    return snippets2
