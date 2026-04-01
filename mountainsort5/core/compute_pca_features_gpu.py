import torch

def compute_pca_features(X: torch.Tensor, *, npca: int) -> torch.Tensor:
    """
    Compute PCA features using pure PyTorch.
    Includes deterministic sign-flipping to match the output of scikit-learn.
    
    Parameters
    ----------
    X : torch.Tensor
        Input data (L x D).
    npca : int
        Desired number of PCA features.
    """
    L = X.shape[0]
    D = X.shape[1]
    npca_2 = min(npca, L, D)

    if L == 0 or D == 0:
        return torch.zeros((0, npca_2), dtype=torch.float32, device=X.device)

    # 1. Center the data (scikit-learn always centers)
    mean = torch.mean(X, dim=0)
    X_centered = X - mean

    # 2. Singular Value Decomposition
    # Using exact SVD (economy mode) to match sklearn's default full solver
    U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)

    # 3. Deterministic Sign Flip
    # Matches scikit-learn's svd_flip ensuring the max absolute value
    # in each column of U is positive.
    max_abs_cols = torch.argmax(torch.abs(U), dim=0)
    signs = torch.sign(U[max_abs_cols, torch.arange(U.shape[1])])
    signs[signs == 0] = 1.0  # Handle edge cases of pure 0
    U *= signs

    # 4. Transform data
    # The projection onto principal components is U * S
    X_transformed = U[:, :npca_2] * S[:npca_2]

    return X_transformed