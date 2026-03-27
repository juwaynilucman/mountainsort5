from typing import Any

def _is_torch_tensor(x: Any) -> bool:
    """Check if an object is a PyTorch tensor without forcing a torch import."""
    return type(x).__module__.startswith('torch') and type(x).__name__ == 'Tensor'

def compute_pca_features(X: Any, **kwargs):
    if _is_torch_tensor(X):
        from .compute_pca_features_gpu import compute_pca_features as compute_pca_features_gpu
        return compute_pca_features_gpu(X, **kwargs)
    else:
        from .compute_pca_features_cpu import compute_pca_features as compute_pca_features_cpu
        return compute_pca_features_cpu(X, **kwargs)