from typing import Any

def _is_torch_tensor(x: Any) -> bool:
    """Check if an object is a PyTorch tensor without forcing a torch import."""
    return type(x).__module__.startswith('torch') and type(x).__name__ == 'Tensor'

def offset_times(times: Any, offsets: Any, labels: Any):
    if _is_torch_tensor(times):
        from .offset_times_gpu import offset_times as offset_times_gpu
        return offset_times_gpu(times, offsets, labels)
    else:
        from .offset_times_cpu import offset_times as offset_times_cpu
        return offset_times_cpu(times, offsets, labels)
