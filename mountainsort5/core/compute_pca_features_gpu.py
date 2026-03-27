import torch

def compute_pca_features(X: torch.Tensor, *, npca: int) -> torch.Tensor:
    """
    Compute PCA features using PyTorch.
    
    Parameters
    ----------
    X : torch.Tensor
        Input data (L x D).
    npca : int
        Number of PCA features to return.

    Returns
    -------
    torch.Tensor
        PCA features (L x npca_2).
    """
    L, D = X.shape
    npca_2 = min(npca, L, D)

    if L == 0 or D == 0:
        return torch.zeros((L, npca_2), device=X.device, dtype=X.dtype)

    # 1. Center the data (Subtract the mean of each feature)
    mean = torch.mean(X, dim=0)
    X_centered = X - mean

    # 2. Compute SVD
    # full_matrices=False gives us the "economy" SVD
    U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)

    # 3. Project the data onto the principal components
    # The columns of U * S represent the principal components
    return (U * S)[:, :npca_2]
