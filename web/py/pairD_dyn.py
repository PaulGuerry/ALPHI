"""UMAP neighbourhood size used by ALPHI for N patients (compute_dynamic_parameters)."""


def n_neighbors_for(n):
    max_neighbors = max(2, n - 1)
    if n <= 15:
        return max(2, n - 2)
    if n <= 50:
        return min(10, max_neighbors)
    return min(30, max_neighbors)
