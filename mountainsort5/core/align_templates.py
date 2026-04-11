from typing import Any

def _is_torch_tensor(x: Any) -> bool:
    """Check if an object is a PyTorch tensor without forcing a torch import."""
    return type(x).__module__.startswith('torch') and type(x).__name__ == 'Tensor'

def align_templates(templates: Any):
    if _is_torch_tensor(templates):
        from .align_templates_gpu import align_templates as align_templates_gpu
        return align_templates_gpu(templates)
    else:
        from .align_templates_cpu import align_templates as align_templates_cpu
        return align_templates_cpu(templates)