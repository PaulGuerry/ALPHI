"""Stand-in for the hdbscan package (Cython, not available in Pyodide).

HDBSCAN: scikit-learn's implementation (sklearn.cluster.HDBSCAN, same algorithm as the
hdbscan package of McInnes et al.: mutual reachability, condensed tree, excess-of-mass
selection), with the hdbscan package's argument names. validity: DBCV (Moulavi et al., 2014),
ported from hdbscan.validity.
"""
import numpy as np
from sklearn.cluster import HDBSCAN as _SkHDBSCAN
from . import validity  # noqa: F401


class HDBSCAN:
    def __init__(self, min_cluster_size=5, min_samples=None, cluster_selection_epsilon=0.0,
                 metric="euclidean", allow_single_cluster=False, cluster_selection_method="eom",
                 gen_min_span_tree=False, **ignored):
        self._kw = dict(min_cluster_size=int(min_cluster_size),
                        min_samples=None if min_samples is None else int(min_samples),
                        cluster_selection_epsilon=float(cluster_selection_epsilon), metric=metric,
                        allow_single_cluster=bool(allow_single_cluster),
                        cluster_selection_method=cluster_selection_method)

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        m = _SkHDBSCAN(copy=True, **self._kw).fit(X)
        self.labels_ = m.labels_
        self.probabilities_ = m.probabilities_
        return self

    def fit_predict(self, X):
        return self.fit(X).labels_
