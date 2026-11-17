"""Stand-in for the parts of lifelines used by ALPHI (lifelines pulls in autograd, formulaic
and matplotlib): Kaplan-Meier estimator and Cox proportional hazards model (Efron ties,
Newton-Raphson, as lifelines' CoxPHFitter without penalty)."""
import numpy as np
import pandas as pd
from scipy import stats


class KaplanMeierFitter:
    def fit(self, durations, event_observed, label=None):
        t = np.asarray(durations, dtype=float)
        e = np.asarray(event_observed, dtype=float)
        times = np.unique(t)
        s, surv = 1.0, []
        for u in times:
            at_risk = np.sum(t >= u)
            d = np.sum((t == u) & (e > 0))
            if at_risk > 0:
                s *= 1.0 - d / at_risk
            surv.append(s)
        self.timeline, self.survival = np.concatenate([[0.0], times]), np.concatenate([[1.0], surv])
        below = np.where(self.survival <= 0.5)[0]
        self.median_survival_time_ = float(self.timeline[below[0]]) if len(below) else np.inf
        return self

    def predict(self, time):
        i = np.searchsorted(self.timeline, time, side="right") - 1
        return float(self.survival[max(i, 0)])


class _LRT:
    def __init__(self, stat, df):
        self.test_statistic = float(stat)
        self.p_value = float(stats.chi2.sf(stat, df))


class CoxPHFitter:
    def fit(self, df, duration_col, event_col, max_iter=50, tol=1e-9):
        cov = [c for c in df.columns if c not in (duration_col, event_col)]
        X = df[cov].to_numpy(dtype=float)
        T = df[duration_col].to_numpy(dtype=float)
        E = df[event_col].to_numpy(dtype=float)
        mu, sd = X.mean(0), X.std(0)
        sd[sd == 0] = 1.0
        Z = (X - mu) / sd                                     # standardized, as lifelines
        beta = np.zeros(Z.shape[1])
        ll0 = self._loglik(Z, T, E, beta)[0]
        ll = ll0
        for _ in range(max_iter):
            ll, g, H = self._loglik(Z, T, E, beta)
            step = np.linalg.solve(-H, g)
            nb = beta + step
            while self._loglik(Z, T, E, nb)[0] < ll - 1e-12 and np.max(np.abs(nb - beta)) > 1e-12:
                nb = beta + (nb - beta) / 2.0                 # step halving
            if np.max(np.abs(nb - beta)) < tol:
                beta = nb
                break
            beta = nb
        ll, g, H = self._loglik(Z, T, E, beta)
        var = np.linalg.inv(-H)
        coef = beta / sd
        se = np.sqrt(np.diag(var)) / sd
        z = coef / se
        zc = stats.norm.ppf(0.975)
        self.summary = pd.DataFrame({
            "coef": coef, "exp(coef)": np.exp(coef), "se(coef)": se,
            "coef lower 95%": coef - zc * se, "coef upper 95%": coef + zc * se,
            "exp(coef) lower 95%": np.exp(coef - zc * se), "exp(coef) upper 95%": np.exp(coef + zc * se),
            "z": z, "p": 2 * stats.norm.sf(np.abs(z))}, index=pd.Index(cov, name="covariate"))
        self.params_ = pd.Series(coef, index=cov)
        self._ll, self._ll0, self._k = ll, ll0, len(cov)
        self.concordance_index_ = _concordance(T, -(X @ coef), E)
        return self

    def log_likelihood_ratio_test(self):
        return _LRT(2 * (self._ll - self._ll0), self._k)

    @staticmethod
    def _loglik(Z, T, E, beta):
        """Efron partial log-likelihood, gradient and Hessian."""
        p = Z.shape[1]
        ll, g, H = 0.0, np.zeros(p), np.zeros((p, p))
        eta = Z @ beta
        w = np.exp(eta)
        for t in np.unique(T[E > 0]):
            risk = T >= t
            dead = (T == t) & (E > 0)
            d = int(dead.sum())
            r0, r1 = w[risk].sum(), (w[risk, None] * Z[risk]).sum(0)
            r2 = (w[risk, None, None] * Z[risk, :, None] * Z[risk, None, :]).sum(0)
            d0, d1 = w[dead].sum(), (w[dead, None] * Z[dead]).sum(0)
            d2 = (w[dead, None, None] * Z[dead, :, None] * Z[dead, None, :]).sum(0)
            ll += eta[dead].sum()
            g += Z[dead].sum(0)
            for l in range(d):
                f = l / d
                s0, s1, s2 = r0 - f * d0, r1 - f * d1, r2 - f * d2
                ll -= np.log(s0)
                g -= s1 / s0
                H -= s2 / s0 - np.outer(s1, s1) / s0 ** 2
        return ll, g, H


def _concordance(T, pred, E):
    """Harrell's C (higher prediction = longer survival), ties in prediction count 1/2."""
    num = den = 0.0
    for i in range(len(T)):
        if not E[i]:
            continue
        later = T > T[i]
        den += later.sum()
        num += (pred[later] > pred[i]).sum() + 0.5 * (pred[later] == pred[i]).sum()
    return num / den if den else np.nan
