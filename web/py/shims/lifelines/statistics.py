"""Log-rank tests (stand-in for lifelines.statistics)."""
import numpy as np
from scipy import stats


class _Result:
    def __init__(self, stat, df):
        self.test_statistic = float(stat)
        self.p_value = float(stats.chi2.sf(stat, df))


def multivariate_logrank_test(event_durations, groups, event_observed=None):
    T = np.asarray(event_durations, dtype=float)
    G = np.asarray(groups)
    E = np.ones_like(T) if event_observed is None else np.asarray(event_observed, dtype=float)
    labs = np.unique(G)
    k = len(labs)
    O, Ex, V = np.zeros(k), np.zeros(k), np.zeros((k, k))
    for t in np.unique(T[E > 0]):
        risk = T >= t
        n = risk.sum()
        d = ((T == t) & (E > 0)).sum()
        nj = np.array([(risk & (G == g)).sum() for g in labs], dtype=float)
        dj = np.array([((T == t) & (E > 0) & (G == g)).sum() for g in labs], dtype=float)
        O += dj
        Ex += d * nj / n
        if n > 1:
            V += d * (n - d) / (n - 1) * (np.diag(nj / n) - np.outer(nj, nj) / n ** 2)
    diff = (O - Ex)[:-1]
    stat = diff @ np.linalg.pinv(V[:-1, :-1]) @ diff
    return _Result(stat, k - 1)


def logrank_test(durations_A, durations_B, event_observed_A=None, event_observed_B=None):
    a, b = np.asarray(durations_A, float), np.asarray(durations_B, float)
    ea = np.ones_like(a) if event_observed_A is None else np.asarray(event_observed_A, float)
    eb = np.ones_like(b) if event_observed_B is None else np.asarray(event_observed_B, float)
    return multivariate_logrank_test(np.concatenate([a, b]), np.array([0] * len(a) + [1] * len(b)),
                                     np.concatenate([ea, eb]))
