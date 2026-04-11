from typing import Any

def compute_templates(snippets: Any, labels: Any):
    """Compute templates from snippets and labels, matching the NumPy reference
    as closely as practical.

    Args:
        snippets: L x T x M torch tensor
        labels: array-like or tensor of length L with labels for the snippets

    Returns:
        templates: K x T x M torch tensor where K is the number of clusters
    """
    import torch

    if not isinstance(snippets, torch.Tensor):
        raise TypeError("snippets must be a torch.Tensor")

    L = snippets.shape[0]
    T = snippets.shape[1]
    M = snippets.shape[2]

    labels_tensor = torch.as_tensor(labels, device=snippets.device)

    if len(labels_tensor) != L:
        raise Exception("Length of labels must equal number of snippets")

    if L == 0:
        return torch.zeros((0, T, M), dtype=torch.float32, device=snippets.device)

    K = int(torch.max(labels_tensor).item())

    templates = torch.zeros((K, T, M), dtype=torch.float32, device=snippets.device)

    for k in range(1, K + 1):
        snippets1 = snippets[labels_tensor == k]

        if snippets1.shape[0] == 0:
            # Match the reference behavior more closely for missing labels
            # instead of silently leaving zeros.
            templates[k - 1] = torch.full(
                (T, M),
                float("nan"),
                dtype=torch.float32,
                device=snippets.device,
            )
        else:
            templates[k - 1] = torch.quantile(snippets1, 0.5, dim=0)

    return templates