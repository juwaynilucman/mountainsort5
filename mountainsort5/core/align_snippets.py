from typing import Any

def _is_torch_tensor(x: Any) -> bool:
    """Check if an object is a PyTorch tensor without forcing a torch import."""
    return type(x).__module__.startswith('torch') and type(x).__name__ == 'Tensor'

def align_snippets(snippets: Any, offsets: Any, labels: Any):
    if _is_torch_tensor(snippets):
        from .align_snippets_gpu import align_snippets as align_snippets_gpu
        return align_snippets_gpu(snippets, offsets, labels)
    else:
        from .align_snippets_cpu import align_snippets as align_snippets_cpu
        return align_snippets_cpu(snippets, offsets, labels)