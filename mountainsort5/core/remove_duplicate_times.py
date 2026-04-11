from typing import Any


def _is_torch_tensor(x: Any) -> bool:
    """Check if an object is a PyTorch tensor without forcing a torch import."""
    return type(x).__module__.startswith('torch') and type(x).__name__ == 'Tensor'


def remove_duplicate_times(times: Any, labels: Any):
    if _is_torch_tensor(times):
        from .remove_duplicate_times_gpu import remove_duplicate_times as remove_duplicate_times_gpu
        return remove_duplicate_times_gpu(times, labels)
    else:
        from .remove_duplicate_times_cpu import remove_duplicate_times as remove_duplicate_times_cpu
        return remove_duplicate_times_cpu(times, labels)
