#!/usr/local/bin/python3.14
"""
Latent class check of the ALPHI clustering (alphi.py --lca).

1. Feature families. The HPO terms reported in the cohort are grouped by complete linkage on
   their distance in the HPO (number of steps through the closest common ancestor), so that
   every two terms of a family are at most --lca-max-step steps apart (default 2: a term,
   its parent and its grandparent, or siblings). A patient has a family when he or she
   reports any of its terms, so that the same feature reported with different terms (by
   different authors, or with a broader term) counts once. Families present in fewer than
   --lca-min-count patients, or missing in fewer, are not used.
2. Latent class analysis (mixture of independent Bernoulli variables, fitted by EM with
   several random starts and one start from the ALPHI clusters) for every number of classes
   K from 1 to 6 and from k_ALPHI - 3 to k_ALPHI + 3; K is chosen by the Bayesian
   information criterion (BIC).
3. Checks of the LCA solution: resampling stability (the model with the chosen K refitted on
   --lca-resamples subsamples of 80% of the patients; each class matched to its most similar
   class by the Jaccard index of members; stable when the mean is >= 0.75), and the
   few-feature check of ALPHI applied to the classes.
4. Decision (the final clustering, exported as the optimal solution):
   - concordant: same number of groups and ARI >= 0.8 on the patients ALPHI assigned:
     ALPHI's clusters are kept;
   - LCA finds more groups: its classes are adopted if all are stable and none is flagged by
     the few-feature check (structure that the density-based clustering merged, e.g. a group
     lying between two others, or sparse reporting); otherwise ALPHI's clusters are kept;
   - LCA finds fewer groups: ALPHI's clusters are kept unless one of them is flagged by the
     few-feature check or is not supported by resampling (mean Jaccard <= 0.5) while the LCA
     classes are stable and unflagged, in which case the classes are adopted;
   - same number of groups but different partitions: ALPHI's clusters are kept.
   The patients placed differently by the two methods are listed in every case, with their
   posterior probability of belonging to each class.
"""
import math

import numpy as np


def norm(code):
    return str(code).replace("HP:", "").zfill(7)


def _ups(code, tree_lookup):
    """{ancestor: number of steps up} for one term (itself at 0)."""
    d = {code: 0}
    entry = tree_lookup.get(code)
    for route in (entry or {}).get("parent_branches", []):
        for i, c in enumerate(route):
            c = norm(c)
            if c != "-------" and (c not in d or i < d[c]):
                d[c] = i
    return d


def feature_families(code_sets, tree_lookup, max_step=2, min_count=None):
    """Returns (families: list of sets of codes, X: patients x families presence matrix)."""
    n = len(code_sets)
    if min_count is None:
        min_count = max(3, math.ceil(0.03 * n))
    terms = sorted(set().union(*code_sets)) if code_sets else []
    ups = {t: _ups(t, tree_lookup) for t in terms}
    m = len(terms)
    D = np.full((m, m), 99.0)
    for i in range(m):
        D[i, i] = 0.0
        ui = ups[terms[i]]
        for j in range(i + 1, m):
            uj = ups[terms[j]]
            common = ui.keys() & uj.keys()
            if common:
                D[i, j] = D[j, i] = min(ui[c] + uj[c] for c in common)
    clusters = [[i] for i in range(m)]
    while True:                                    # complete linkage, merge closest first
        best = None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                d = D[np.ix_(clusters[a], clusters[b])].max()
                if d <= max_step and (best is None or d < best[0]):
                    best = (d, a, b)
        if best is None:
            break
        _, a, b = best
        clusters[a] += clusters[b]
        del clusters[b]
    fams = [{terms[i] for i in c} for c in clusters]
    X = np.array([[bool(cs & f) for f in fams] for cs in code_sets], dtype=float)
    keep = [j for j in range(len(fams)) if min_count <= X[:, j].sum() <= n - min_count]
    return [fams[j] for j in keep], X[:, keep]


def fit_lca(X, K, restarts=10, iters=300, seed=0, init_labels=None, tol=1e-6):
    """EM for a mixture of independent Bernoulli variables (Beta(1,1) smoothing).
    Returns (bic, loglik, labels, posterior, theta, pi)."""
    n, d = X.shape
    rng = np.random.default_rng(seed)
    starts = []
    if init_labels is not None:
        lab = np.asarray(init_labels)
        u = [c for c in sorted(set(lab)) if c != -1][:K]
        if len(u) == K:
            R = np.full((n, K), 1e-3)
            for k, c in enumerate(u):
                R[lab == c, k] = 1.0
            R[lab == -1] = 1.0 / K
            starts.append(R / R.sum(1, keepdims=True))
    starts += [rng.dirichlet(np.ones(K), size=n) for _ in range(restarts)]
    best = None
    for R in starts:
        prev = -np.inf
        for _ in range(iters):
            pi = R.mean(0) + 1e-12
            th = (R.T @ X + 1.0) / (R.sum(0)[:, None] + 2.0)
            L = X @ np.log(th).T + (1 - X) @ np.log(1 - th).T + np.log(pi)
            mx = L.max(1, keepdims=True)
            E = np.exp(L - mx)
            ll = float((np.log(E.sum(1)) + mx[:, 0]).sum())
            R = E / E.sum(1, keepdims=True)
            if ll - prev < tol:
                break
            prev = ll
        if best is None or ll > best[0]:
            best = (ll, R, th, pi)
    ll, R, th, pi = best
    n_par = K * d + (K - 1)
    return -2 * ll + n_par * math.log(n), ll, R.argmax(1), R, th, pi


def ari(a, b):
    from collections import Counter
    from math import comb
    n = len(a)
    if n < 2:
        return float("nan")
    cont, ra, rb = Counter(zip(a, b)), Counter(a), Counter(b)
    s = sum(comb(v, 2) for v in cont.values())
    sa, sb = sum(comb(v, 2) for v in ra.values()), sum(comb(v, 2) for v in rb.values())
    exp = sa * sb / comb(n, 2)
    mx = (sa + sb) / 2
    return 1.0 if mx == exp else (s - exp) / (mx - exp)


def lca_stability(X, K, labels, n_resamples=20, fraction=0.8, seed=1, restarts=None):
    """Mean Jaccard recovery of each class over refits on subsamples."""
    n = len(labels)
    m = max(2, int(round(fraction * n)))
    rng = np.random.default_rng(seed)
    classes = sorted(set(labels))
    jac = {c: [] for c in classes}
    for b in range(n_resamples):
        idx = np.sort(rng.choice(n, size=m, replace=False))
        _, _, sub, _, _, _ = fit_lca(X[idx], K, restarts=restarts if restarts is not None else (5 if K <= 10 else 1),
                                     seed=100 + b, init_labels=labels[idx])
        found = [set(idx[sub == k]) for k in set(sub)]
        for c in classes:
            orig = set(idx[labels[idx] == c])
            if orig:
                jac[c].append(max((len(orig & f) / len(orig | f) for f in found), default=0.0))
    return {int(c): float(np.mean(v)) if v else float("nan") for c, v in jac.items()}


def check(code_sets, tree_lookup, alphi_labels, flag_fn, alphi_flags=None, alphi_stability=None,
          max_step=2, min_count=None, restarts=10, n_resamples=20, seed=0):
    """Runs the latent class check. flag_fn(labels) -> few-feature flags {cluster: {"flag", ...}}
    (ALPHI's own check). Returns a report dict with the final ("hybrid") labels."""
    alphi = np.asarray(alphi_labels)
    k_a = len(set(alphi) - {-1})
    fams, X = feature_families(code_sets, tree_lookup, max_step, min_count)
    rep = dict(n_families=len(fams), k_alphi=k_a, families=[sorted(f) for f in fams])
    if X.shape[1] < 2:
        rep.update(k_lca=1, decision="no usable features: ALPHI kept", adopted="ALPHI",
                   hybrid_labels=alphi.tolist(), disagreements=[])
        return rep
    # K from 1 to 6, and around ALPHI's number of clusters (k_ALPHI - 3 to k_ALPHI + 3)
    ks = sorted(set(range(1, 7)) | set(range(max(1, k_a - 3), k_a + 4)))
    fits = {}
    for K in ks:
        fits[K] = fit_lca(X, K, restarts=restarts if K <= 10 else 3, seed=seed + K,
                          init_labels=alphi if K == k_a else None)
    bic = {K: f[0] for K, f in fits.items()}
    k_l = min(bic, key=bic.get)
    _, _, lab, post, _, _ = fits[k_l]
    rep.update(bic={str(K): round(v, 1) for K, v in bic.items()}, k_lca=k_l,
               class_sizes=[int((lab == c).sum()) for c in range(k_l)])
    assigned = alphi != -1
    agreement = ari(alphi[assigned].tolist(), lab[assigned].tolist()) if assigned.sum() > 1 else float("nan")
    rep["ari_alphi_lca"] = agreement
    stab = lca_stability(X, k_l, lab, n_resamples=n_resamples) if k_l > 1 else {0: 1.0}
    flags = flag_fn(lab) if k_l > 1 else {}
    lca_ok = all(v >= 0.75 for v in stab.values()) and not any(v["flag"] for v in flags.values())
    rep.update(lca_stability=stab, lca_flagged=[c for c, v in flags.items() if v["flag"]], lca_valid=bool(lca_ok))
    a_flag = any(v["flag"] for v in (alphi_flags or {}).values())
    a_unsupported = any(v.get("jaccard_mean", 1) <= 0.5 for v in (alphi_stability or {}).values())
    if k_l == k_a and agreement >= 0.8:
        decision, adopted = "concordant: ALPHI kept", "ALPHI"
    elif k_l > k_a:
        if lca_ok:
            decision, adopted = "LCA finds more groups, stable and unflagged: LCA classes adopted", "LCA"
        else:
            decision, adopted = "LCA finds more groups, not supported by its checks: ALPHI kept", "ALPHI"
    elif k_l < k_a:
        if (a_flag or a_unsupported) and lca_ok:
            decision, adopted = ("LCA finds fewer groups and ALPHI has a flagged or unsupported cluster: "
                                 "LCA classes adopted"), "LCA"
        else:
            decision, adopted = "LCA finds fewer groups: ALPHI kept", "ALPHI"
    else:
        decision, adopted = "same number of groups, different partitions: ALPHI kept", "ALPHI"
    hybrid = lab if adopted == "LCA" else alphi
    rep.update(decision=decision, adopted=adopted, hybrid_labels=[int(x) for x in hybrid])
    # patients placed differently: map each LCA class to the ALPHI cluster it overlaps most
    mapping = {}
    for c in range(k_l):
        a = alphi[(lab == c) & assigned]
        if len(a):
            vals, cnt = np.unique(a, return_counts=True)
            mapping[c] = int(vals[cnt.argmax()])
    dis = []
    for i in range(len(alphi)):
        if alphi[i] != -1 and mapping.get(int(lab[i])) is not None and mapping[int(lab[i])] != alphi[i]:
            dis.append(dict(index=i, alphi_cluster=int(alphi[i]), lca_class=int(lab[i]),
                            lca_class_like_alphi_cluster=mapping[int(lab[i])],
                            posterior=[round(float(p), 3) for p in post[i]]))
    rep["disagreements"] = dis
    rep["posterior"] = [[round(float(p), 3) for p in row] for row in post]
    return rep
