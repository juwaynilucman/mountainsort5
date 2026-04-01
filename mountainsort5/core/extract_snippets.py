from typing import Any

def _is_torch_tensor(x: Any) -> bool:
    """Check if an object is a PyTorch tensor without forcing a torch import."""
    return type(x).__module__.startswith('torch') and type(x).__name__ == 'Tensor'


def extract_snippets(
    traces: Any,
    **kwargs):
    if _is_torch_tensor(traces):
        from .extract_snippets_gpu import extract_snippets as extract_snippets_gpu
        return extract_snippets_gpu(traces, **kwargs)
    else:
        from .extract_snippets_cpu import extract_snippets as extract_snippets_cpu
        return extract_snippets_cpu(traces, **kwargs)

def extract_snippets_in_channel_neighborhood(
    traces: Any,
    **kwargs):
    if _is_torch_tensor(traces):
        from .extract_snippets_gpu import extract_snippets_in_channel_neighborhood as extract_snippets_in_channel_neighborhood_gpu
        return extract_snippets_in_channel_neighborhood_gpu(traces, **kwargs)
    else:
        from .extract_snippets_cpu import extract_snippets_in_channel_neighborhood as extract_snippets_in_channel_neighborhood_cpu
        return extract_snippets_in_channel_neighborhood_cpu(traces, **kwargs)
