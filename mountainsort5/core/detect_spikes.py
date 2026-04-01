from typing import Any



def _is_torch_tensor(x: Any) -> bool:
    """Check if an object is a PyTorch tensor without forcing a torch import."""
    return type(x).__module__.startswith('torch') and type(x).__name__ == 'Tensor'

def detect_spikes(
    traces: Any,
    **kwargs):
    if _is_torch_tensor(traces):
        # Lazy import to avoid making PyTorch a hard dependency
        from .detect_spikes_gpu import detect_spikes as detect_spikes_gpu
        return detect_spikes_gpu(traces, **kwargs)
    else:
        from .detect_spikes_cpu import detect_spikes as detect_spikes_cpu
        return detect_spikes_cpu(traces, **kwargs)