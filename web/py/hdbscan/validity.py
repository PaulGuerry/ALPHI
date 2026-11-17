"""Density-based cluster validation (DBCV), port of hdbscan.validity (hdbscan package)."""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances


def mst_linkage_core(distance_matrix):
    """Prim's algorithm on a dense matrix, as hdbscan._hdbscan_linkage.mst_linkage_core."""
    n = distance_matrix.shape[0]
    result = np.zeros((n - 1, 3))
    node_labels = np.arange(n, dtype=np.intp)
    current_node = 0
    current_distances = np.full(n, np.inf)
    current_labels = node_labels
    for i in range(1, n):
        label_filter = current_labels != current_node
        current_labels = current_labels[label_filter]
        left = current_distances[label_filter]
        right = distance_matrix[current_node][current_labels]
        current_distances = np.where(left < right, left, right)
        new_node_index = np.argmin(current_distances)
        new_node = current_labels[new_node_index]
        result[i - 1, 0] = current_node
        result[i - 1, 1] = new_node
        result[i - 1, 2] = current_distances[new_node_index]
        current_node = new_node
    return result


def all_points_core_distance(distance_matrix, d=2.0):
    nz = distance_matrix != 0
    distance_matrix[nz] = (1.0 / distance_matrix[nz]) ** d
    result = distance_matrix.sum(axis=1)
    result /= distance_matrix.shape[0] - 1
    if result.sum() == 0:
        return np.zeros(len(distance_matrix))
    return result ** (-1.0 / d)


def distances_between_points(X, labels, cluster_id, metric="euclidean", d=None, no_coredist=False):
    if metric == "precomputed":
        distance_matrix = X[labels == cluster_id, :][:, labels == cluster_id]
    else:
        distance_matrix = pairwise_distances(X[labels == cluster_id, :], metric=metric)
        d = X.shape[1]
    if no_coredist:
        return distance_matrix, None
    core = all_points_core_distance(distance_matrix.copy(), d=d)
    core_i = np.tile(core, (core.shape[0], 1))
    return np.dstack([distance_matrix, core_i, core_i.T]).max(axis=-1), core


def internal_minimum_spanning_tree(mr_distances):
    single_linkage_data = mst_linkage_core(mr_distances)
    min_span_tree = single_linkage_data.copy()
    for index, row in enumerate(min_span_tree[1:], 1):
        candidates = np.where(np.isclose(mr_distances[int(row[1])], row[2]))[0]
        candidates = np.intersect1d(candidates, single_linkage_data[:index, :2].astype(int))
        candidates = candidates[candidates != row[1]]
        if len(candidates):
            row[0] = candidates[0]
    vertices = np.arange(mr_distances.shape[0])[np.bincount(min_span_tree.T[:2].flatten().astype(np.intp)) > 1]
    edge_selection = np.prod(np.isin(min_span_tree.T[:2], vertices).reshape((min_span_tree.shape[0], 2), order="F"),
                             axis=1).astype(bool)
    edges = min_span_tree[edge_selection] if np.any(edge_selection) else min_span_tree.copy()
    return vertices, edges


def density_separation(X, labels, c1, c2, nodes1, nodes2, core1, core2, metric="euclidean", no_coredist=False):
    if metric == "precomputed":
        dm = X[labels == c1, :][:, labels == c2][nodes1, :][:, nodes2]
    else:
        dm = cdist(X[labels == c1][nodes1], X[labels == c2][nodes2], metric)
    if no_coredist:
        return dm.min()
    cm1 = np.tile(core1[nodes1], (dm.shape[1], 1)).T
    cm2 = np.tile(core2[nodes2], (dm.shape[0], 1))
    return np.dstack([dm, cm1, cm2]).max(axis=-1).min()


def validity_index(X, labels, metric="euclidean", d=None, per_cluster_scores=False, mst_raw_dist=False,
                   verbose=False):
    core_distances, density_sparseness, mst_nodes = {}, {}, {}
    max_cluster_id = labels.max() + 1
    density_sep = np.inf * np.ones((max_cluster_id, max_cluster_id), dtype=np.float64)
    cvi = np.empty(max_cluster_id, dtype=np.float64)
    for c in range(max_cluster_id):
        if np.sum(labels == c) == 0:
            continue
        dist_mst, core_distances[c] = distances_between_points(X, labels, c, metric, d, no_coredist=mst_raw_dist)
        mst_nodes[c], edges = internal_minimum_spanning_tree(dist_mst)
        density_sparseness[c] = edges.T[2].max()
    for i in range(max_cluster_id):
        if np.sum(labels == i) == 0:
            continue
        for j in range(i + 1, max_cluster_id):
            if np.sum(labels == j) == 0:
                continue
            density_sep[i, j] = density_sep[j, i] = density_separation(
                X, labels, i, j, mst_nodes[i], mst_nodes[j], core_distances[i], core_distances[j],
                metric=metric, no_coredist=mst_raw_dist)
    n = float(X.shape[0])
    result = 0.0
    for i in range(max_cluster_id):
        if np.sum(labels == i) == 0:
            continue
        mds = density_sep[i].min()
        cvi[i] = (mds - density_sparseness[i]) / max(mds, density_sparseness[i])
        result += (np.sum(labels == i) / n) * cvi[i]
    return (result, cvi) if per_cluster_scores else result
