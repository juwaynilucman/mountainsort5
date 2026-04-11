import torch


def remove_duplicate_times(times: torch.Tensor, labels):
    if len(times) == 0:
        return times, labels
    inds = torch.where(torch.diff(times) > 0)[0]
    inds = torch.cat([torch.tensor([0], device=times.device), inds + 1])
    return times[inds], labels[inds]
