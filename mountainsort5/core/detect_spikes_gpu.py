import numpy as np
import numpy.typing as npt
from typing import Tuple, Union
import torch
import torch.nn.functional as F

def detect_spikes(
    traces: npt.NDArray[np.float32],
    *,
    channel_locations: npt.NDArray[np.float32],
    time_radius: int,
    channel_radius: Union[float, None],
    detect_threshold: float,
    detect_sign: int,
    margin_left: int,
    margin_right: int,
    verbose: bool = False
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.int32]]:
    """
    GPU-accelerated spike detection using PyTorch.
    
    This function replicates the exact behavior of `mountainsort5.core.detect_spikes.detect_spikes`,
    but utilizes GPU arrays for faster thresholding and peak finding.
    """
    if verbose:
        print("Running detect_spikes on GPU using PyTorch...")
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    N, M = traces.shape
    # traces_pt = torch.tensor(traces, device=device)
    traces_pt = traces.to(device)
    
    if detect_sign > 0:
        traces_pt = -traces_pt
    elif detect_sign == 0:
        traces_pt = -torch.abs(traces_pt)
        
    # Invert traces so we are looking for local maxima above detect_threshold
    traces_inv = -traces_pt
    
    # 1. Temporal Max Pooling
    pool_kernel = 2 * time_radius + 1
    # max_pool1d expects (batch, channels, length) -> (1, M, N)
    traces_inv_t = traces_inv.t().unsqueeze(0)
    
    temporal_max = F.max_pool1d(
        traces_inv_t,
        kernel_size=pool_kernel,
        stride=1,
        padding=time_radius
    ).squeeze(0).t() # -> (N, M)
    
    # Identify candidates
    is_temporal_peak = (traces_inv == temporal_max)
    is_suprathreshold = (traces_inv >= detect_threshold)
    valid_mask = is_temporal_peak & is_suprathreshold
    
    # Apply margins
    if margin_left > 0:
        valid_mask[:margin_left, :] = False
    if margin_right > 0:
        valid_mask[N - margin_right:, :] = False
        
    # 2. Spatial check
    locs = torch.tensor(channel_locations, device=device)
    if channel_radius is not None:
        dists = torch.cdist(locs, locs)
        adj_matrix = dists <= channel_radius
    else:
        adj_matrix = torch.ones((M, M), dtype=torch.bool, device=device)
        
    # For each channel, find the max in its neighborhood
    spatial_max = torch.empty_like(temporal_max)
    for m in range(M):
        neighbors = adj_matrix[m]
        spatial_max[:, m] = temporal_max[:, neighbors].max(dim=1).values
        
    is_spatial_peak = (traces_inv >= spatial_max)
    valid_mask = valid_mask & is_spatial_peak
    
    # Extract times and channels
    times_pt, channels_pt = torch.nonzero(valid_mask, as_tuple=True)
    
    times = times_pt.to(torch.int32)
    channel_indices = channels_pt.to(torch.int32)
    
    # Sort
    if len(times) > 0:

        inds = torch.argsort(times, stable=True)
        times = times[inds]
        channel_indices = channel_indices[inds]
        
    return times.to(torch.int64), channel_indices