from typing import Any


def _is_torch_tensor(x: Any) -> bool:
    """Check if an object is a PyTorch tensor without forcing a torch import."""
    return type(x).__module__.startswith('torch') and type(x).__name__ == 'Tensor'


def determine_offsets_to_peak(templates: Any, *, detect_sign: int, T1: int):
    if _is_torch_tensor(templates):
        from .determine_offsets_to_peak_gpu import determine_offsets_to_peak as determine_offsets_to_peak_gpu
        return determine_offsets_to_peak_gpu(templates, detect_sign=detect_sign, T1=T1)
    else:
        from .determine_offsets_to_peak_cpu import determine_offsets_to_peak as determine_offsets_to_peak_cpu
        return determine_offsets_to_peak_cpu(templates, detect_sign=detect_sign, T1=T1)
