import torch
from typing import Union, List, Any

def extract_snippets(
    traces: Any, *,
    channel_locations: Any,
    mask_radius: Union[float, None],
    times: Any,
    channel_indices: Any,
    T1: int,
    T2: int
) -> Any:
    M = traces.shape[1]
    L = times.shape[0]

    if L == 0:
        return torch.zeros((L, T1 + T2, M), dtype=traces.dtype, device=traces.device)

    # Create an (L, T1+T2) matrix of exact extraction indices
    window_indices = torch.arange(-T1, T2, device=traces.device)
    extract_indices = times.unsqueeze(1) + window_indices.unsqueeze(0)

    # Advanced indexing to pull all snippets at once -> (L, T1+T2, M)
    snippets = traces[extract_indices]

    if mask_radius is not None:
        locs = torch.as_tensor(channel_locations, device=traces.device, dtype=torch.float32)
        dists = torch.cdist(locs, locs)
        adj_matrix = dists <= mask_radius
        
        # Mask out channels that aren't within the radius of the peak channel
        valid_channels = adj_matrix[channel_indices]
        valid_mask = valid_channels.unsqueeze(1).to(snippets.dtype)
        
        snippets *= valid_mask # In-place multiplication to save VRAM

    return snippets


def extract_snippets_in_channel_neighborhood(
    traces: Any, *,
    times: Any,
    neighborhood: Union[List[int], None],
    T1: int,
    T2: int
) -> Any:
    L = times.shape[0]

    if neighborhood is None:
        traces_sub = traces
    else:
        traces_sub = traces[:, neighborhood]

    if L == 0:
        return torch.zeros((L, T1 + T2, traces_sub.shape[1]), dtype=traces.dtype, device=traces.device)

    window_indices = torch.arange(-T1, T2, device=traces.device)
    extract_indices = times.unsqueeze(1) + window_indices.unsqueeze(0)

    return traces_sub[extract_indices]
