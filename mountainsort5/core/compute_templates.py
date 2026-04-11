from typing import Any

def _is_torch_tensor(x: Any) -> bool:
    """Check if an object is a PyTorch tensor without forcing a torch import."""
    return type(x).__module__.startswith('torch') and type(x).__name__ == 'Tensor'

def compute_templates(snippets: Any, labels: Any):
    if _is_torch_tensor(snippets):
        from .compute_templates_gpu import compute_templates as compute_templates_gpu
        return compute_templates_gpu(snippets, labels)
    else:
        from .compute_templates_cpu import compute_templates as compute_templates_cpu
        return compute_templates_cpu(snippets, labels)
