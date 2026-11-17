"""Numba-free UMAP for precomputed distance matrices (the only use in ALPHI).

Port of umap-learn 0.5's algorithm (McInnes et al., 2018) for metric="precomputed" on small
data: exact k nearest neighbours from the matrix, smooth_knn_dist, fuzzy simplicial set
(fuzzy union), spectral initialisation (with umap's multi-component layout), and the
sequential SGD layout optimisation, run by js/umap_sgd.js (in the browser through Pyodide,
in CPython through Node). Same defaults as umap-learn: spread 1, local_connectivity 1,
set_op_mix_ratio 1, 500 epochs (N <= 10,000), negative_sample_rate 5, learning rate 1.
Not bit-identical to umap-learn (different random streams, float order): equivalent in
distribution, checked on the clusterings it leads to.

Interface: UMAP(n_components, metric="precomputed", n_neighbors, min_dist,
random_state).fit_transform(D) -> float32 array (N, n_components).
"""
import json, os, subprocess, sys, tempfile
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy.optimize import curve_fit
from scipy.sparse.csgraph import connected_components

SMOOTH_K_TOLERANCE = 1e-5
MIN_K_DIST_SCALE = 1e-3
INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1
_JS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "js", "umap_sgd.js")


def find_ab_params(spread, min_dist):
    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))
    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, _ = curve_fit(curve, xv, yv)
    return float(params[0]), float(params[1])


def smooth_knn_dist(distances, k, n_iter=64, local_connectivity=1.0, bandwidth=1.0):
    target = np.log2(k) * bandwidth
    n = distances.shape[0]
    rho = np.zeros(n, dtype=np.float32)
    result = np.zeros(n, dtype=np.float32)
    mean_distances = np.mean(distances)
    for i in range(n):
        lo, hi, mid = 0.0, np.inf, 1.0
        ith = distances[i]
        nz = ith[ith > 0.0]
        if nz.shape[0] >= local_connectivity:
            index = int(np.floor(local_connectivity))
            interpolation = local_connectivity - index
            if index > 0:
                rho[i] = nz[index - 1]
                if interpolation > SMOOTH_K_TOLERANCE:
                    rho[i] += interpolation * (nz[index] - nz[index - 1])
            else:
                rho[i] = interpolation * nz[0]
        elif nz.shape[0] > 0:
            rho[i] = np.max(nz)
        d = ith[1:].astype(np.float64) - rho[i]
        for _ in range(n_iter):
            psum = float(np.sum(np.where(d > 0, np.exp(-(np.maximum(d, 0) / mid)), 1.0)))
            if abs(psum - target) < SMOOTH_K_TOLERANCE:
                break
            if psum > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                mid = mid * 2 if hi == np.inf else (lo + hi) / 2.0
        result[i] = mid
        if rho[i] > 0.0:
            m = np.mean(ith)
            if result[i] < MIN_K_DIST_SCALE * m:
                result[i] = MIN_K_DIST_SCALE * m
        elif result[i] < MIN_K_DIST_SCALE * mean_distances:
            result[i] = MIN_K_DIST_SCALE * mean_distances
    return result, rho


def fuzzy_simplicial_set(D, n_neighbors, local_connectivity=1.0, set_op_mix_ratio=1.0):
    n = D.shape[0]
    knn_indices = np.argsort(D, axis=1, kind="quicksort")[:, :n_neighbors]
    knn_dists = D[np.arange(n)[:, None], knn_indices].astype(np.float32)
    sigmas, rhos = smooth_knn_dist(knn_dists, float(n_neighbors), local_connectivity=float(local_connectivity))
    rows = np.repeat(np.arange(n), n_neighbors)
    cols = knn_indices.ravel()
    delta = knn_dists - rhos[:, None]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        vals = np.where((delta <= 0) | (sigmas[:, None] == 0), 1.0, np.exp(-(delta / sigmas[:, None])))
    vals[knn_indices == np.arange(n)[:, None]] = 0.0
    result = scipy.sparse.coo_matrix((vals.ravel().astype(np.float32), (rows, cols)), shape=(n, n))
    result.eliminate_zeros()
    transpose = result.transpose()
    prod = result.multiply(transpose)
    result = set_op_mix_ratio * (result + transpose - prod) + (1.0 - set_op_mix_ratio) * prod
    result = scipy.sparse.coo_matrix(result)
    result.eliminate_zeros()
    return result


def _eig_layout(graph, dim):
    diag = np.asarray(graph.sum(axis=0)).ravel()
    n = graph.shape[0]
    I = scipy.sparse.identity(n, dtype=np.float64)
    Dm = scipy.sparse.spdiags(1.0 / np.sqrt(diag), 0, n, n)
    L = I - Dm @ graph @ Dm
    k = dim + 1
    ncv = max(2 * k + 1, int(np.sqrt(n)))
    vals, vecs = scipy.sparse.linalg.eigsh(L, k, which="SM", ncv=ncv, tol=1e-4, v0=np.ones(n), maxiter=n * 5)
    order = np.argsort(vals)[1:k]
    return vecs[:, order]


def _component_layout(D, n_comp, labels, dim, random_state):
    dm = np.zeros((n_comp, n_comp))
    for ci in range(n_comp):
        di = D[labels == ci]
        for cj in range(ci + 1, n_comp):
            dm[ci, cj] = dm[cj, ci] = np.mean(di[:, labels == cj])       # average linkage
    from sklearn.manifold import SpectralEmbedding
    emb = SpectralEmbedding(n_components=dim, affinity="precomputed",
                            random_state=random_state).fit_transform(np.exp(-(dm ** 2)))
    return emb / emb.max()


def spectral_layout(D, graph, dim, random_state):
    n_comp, labels = connected_components(graph)
    if n_comp == 1:
        try:
            return _eig_layout(graph, dim)
        except scipy.sparse.linalg.ArpackError:
            return random_state.uniform(low=-10.0, high=10.0, size=(graph.shape[0], dim))
    # multi-component layout (umap.spectral.multi_component_layout)
    result = np.empty((graph.shape[0], dim), dtype=np.float32)
    if n_comp > 2 * dim:
        meta = _component_layout(D, n_comp, labels, dim, random_state)
    else:
        k = int(np.ceil(n_comp / 2.0))
        base = np.hstack([np.eye(k), np.zeros((k, dim - k))])
        meta = np.vstack([base, -base])[:n_comp]
    csr = graph.tocsr()
    for lab in range(n_comp):
        sel = labels == lab
        cg = csr[sel][:, sel].tocoo()
        dist = np.sqrt(((meta - meta[lab]) ** 2).sum(1))
        data_range = dist[dist > 0.0].min() / 2.0
        m = cg.shape[0]
        if m < 2 * dim or m <= dim + 1:
            result[sel] = random_state.uniform(low=-data_range, high=data_range, size=(m, dim)) + meta[lab]
            continue
        try:
            ce = _eig_layout(cg, dim)
            ce *= data_range / np.max(np.abs(ce))
            result[sel] = ce + meta[lab]
        except scipy.sparse.linalg.ArpackError:
            result[sel] = random_state.uniform(low=-data_range, high=data_range, size=(m, dim)) + meta[lab]
    return result


def make_epochs_per_sample(weights, n_epochs):
    result = -1.0 * np.ones(weights.shape[0], dtype=np.float64)
    n_samples = n_epochs * (weights / weights.max())
    result[n_samples > 0] = float(n_epochs) / np.float64(n_samples[n_samples > 0])
    return result


def _optimize(embedding, head, tail, eps, n_vertices, n_epochs, a, b, seeds):
    """Run js/umap_sgd.js: through the js module in Pyodide, through Node otherwise."""
    dim = embedding.shape[1]
    if sys.platform == "emscripten":
        import js
        def typed(arr, T, dtype):
            return T.new(js.Uint8Array.new(np.ascontiguousarray(arr, dtype=dtype).tobytes()).buffer)
        emb = typed(embedding, js.Float32Array, np.float32)
        js.umapOptimizeLayout(emb, typed(head, js.Int32Array, np.int32), typed(tail, js.Int32Array, np.int32),
                              typed(eps, js.Float64Array, np.float64), int(n_vertices), int(dim), int(n_epochs),
                              float(a), float(b), typed(seeds, js.Uint32Array, np.uint32), 1.0, 1.0, 5)
        return np.frombuffer(js.Uint8Array.new(emb.buffer).to_bytes(), dtype=np.float32).reshape(-1, dim).copy()
    with tempfile.TemporaryDirectory() as t:
        f = lambda name: os.path.join(t, name)
        embedding.astype(np.float32).tofile(f("emb")); head.astype(np.int32).tofile(f("head"))
        tail.astype(np.int32).tofile(f("tail")); eps.astype(np.float64).tofile(f("eps"))
        seeds.astype(np.uint32).tofile(f("seeds"))
        req = dict(embedding=f("emb"), head=f("head"), tail=f("tail"), eps=f("eps"), seeds=f("seeds"),
                   out=f("out"), n_vertices=int(n_vertices), dim=int(dim), n_epochs=int(n_epochs),
                   a=a, b=b, gamma=1.0, initial_alpha=1.0, negative_sample_rate=5)
        json.dump(req, open(f("req.json"), "w"))
        subprocess.run(["node", _JS, f("req.json")], check=True)
        return np.fromfile(f("out"), dtype=np.float32).reshape(-1, dim)


class UMAP:
    def __init__(self, n_components=2, metric="precomputed", n_neighbors=15, min_dist=0.1,
                 random_state=None, spread=1.0, n_epochs=None, **ignored):
        if metric != "precomputed":
            raise ValueError("umap_lite supports metric='precomputed' only")
        self.n_components, self.n_neighbors, self.min_dist = n_components, n_neighbors, min_dist
        self.random_state, self.spread, self.n_epochs = random_state, spread, n_epochs

    def fit_transform(self, X):
        D = np.asarray(X, dtype=np.float64)
        n, dim = D.shape[0], self.n_components
        rs = np.random.RandomState(self.random_state) if not isinstance(self.random_state, np.random.RandomState) \
            else self.random_state
        a, b = find_ab_params(self.spread, self.min_dist)
        graph = fuzzy_simplicial_set(D, self.n_neighbors).tocoo()
        graph.sum_duplicates()
        n_epochs = self.n_epochs or (500 if n <= 10000 else 200)
        graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
        graph.eliminate_zeros()
        init = spectral_layout(D, graph, dim, rs)
        expansion = 10.0 / np.abs(init).max()
        emb = (init * expansion).astype(np.float32) + rs.normal(scale=0.0001, size=[n, dim]).astype(np.float32)
        eps = make_epochs_per_sample(graph.data, n_epochs)
        rng_state = rs.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)
        emb = (10.0 * (emb - emb.min(0)) / (emb.max(0) - emb.min(0))).astype(np.float32)
        # per-vertex random streams (umap-learn 0.5: rng_state + bits of the first coordinate)
        bits = emb[:, 0].astype(np.float64).view(np.int64)
        seeds = ((rng_state[None, :] + bits[:, None]) & 0xFFFFFFFF).astype(np.uint32)
        seeds[seeds < 128] += 128                                     # Tausworthe needs non-trivial states
        self.embedding_ = _optimize(emb, graph.row.astype(np.int32), graph.col.astype(np.int32), eps,
                                    n, n_epochs, a, b, seeds)
        self.graph_ = graph
        return self.embedding_
