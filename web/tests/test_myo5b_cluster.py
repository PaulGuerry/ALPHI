#!/usr/local/bin/python3.14
"""Clustering step on the reference MYO5B distance matrix (cdm.csv of the 11 Nov ALPHI run):
umap_lite + sklearn HDBSCAN + DBCV against the umap-learn + hdbscan results."""
import sys, time
import numpy as np, pandas as pd
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score
sys.path.insert(0, "py")
from umap_lite import UMAP
import hdbscan
REF = sys.argv[1]
cdm = pd.read_csv(f"{REF}/cdm.csv")
X = squareform(cdm["d"].to_numpy())
ref = pd.read_csv(f"{REF}/pairD_UMAP.csv")
for seed in [25, 1, 2, 3, 4]:
    t = time.time()
    emb = UMAP(n_components=10, metric="precomputed", n_neighbors=30, min_dist=0.0, random_state=seed).fit_transform(X)
    tu = time.time() - t
    out = []
    for mcs, ms in [(5, 5), (10, 10), (20, 20), (25, 20), (25, 25)]:
        lab = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, allow_single_cluster=True).fit_predict(emb)
        np.random.seed(42)
        dbcv = hdbscan.validity.validity_index(emb.astype(np.float64) + np.random.normal(0, 1e-7, emb.shape), lab) \
            if len(set(lab) - {-1}) > 1 else np.nan
        ari = adjusted_rand_score(ref["HDBSCAN_mcs5_ms5"], lab)
        sizes = sorted(np.bincount(lab[lab >= 0]), reverse=True)
        out.append(f"mcs{mcs}/ms{ms}: k={len(sizes)} {sizes} noise={np.sum(lab<0)} DBCV={dbcv:.3f} ARI={ari:.3f}")
    print(f"seed {seed} (UMAP {tu:.1f} s)\n  " + "\n  ".join(out))
