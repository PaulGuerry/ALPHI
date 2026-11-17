#!/usr/local/bin/python3.14
"""
Selection of co-occurring terms for ALPHI (alphi.py --select-terms).

Rationale. Latent subgroups are defined by features that occur together in some patients and
not in others; features shared by the whole cohort, or reported independently of one another,
carry no information on the subgroups but dominate the distances when the distinguishing
features are few or incompletely reported. The cohort itself is used to tell them apart:
a term is selected when it co-occurs with other terms more than expected by chance.

Method.
  1. Features. Every term reported in the cohort defines a feature; a patient has the feature
     when he or she reports the term, its parent or one of its children (one step in the
     HPO), so that a feature reported with a substitute term still counts. Features present
     in fewer than --select-min-count patients, or missing in fewer than that number, are
     not tested.
  2. Statistic. Q(t) = N x sum of squared phi coefficients between feature t and every other
     feature, excluding features whose term sets overlap with t's (a term, its parent and
     its children, and siblings through their parent), which co-occur by construction.
     Under independence each N.phi^2 is approximately chi-square with 1 degree of freedom.
  3. Null distribution. The patients are permuted independently for each reported term (which
     keeps the frequency of every term and removes all co-occurrence between terms), the
     features are rebuilt from the permuted terms, so that the overlaps created by the
     features themselves are kept, Q is recomputed for every feature, and the maximum over the
     features is recorded. A feature is selected when its Q
     exceeds the (1 - alpha) quantile of this maximum, which bounds the probability of
     selecting any feature in a cohort without co-occurrence at alpha (family-wise).
  4. Weights. Reported terms that belong to a selected feature (the term, its parent or a
     child) receive weight 1, the others --select-weight (default 0.01), and the distances
     between patients use these weights (weighted mean of the best-match distances for OB,
     weighted Jaccard, weighted best-match average for Resnik). With fewer than two selected
     features, all weights are 1 and ALPHI is unchanged.
"""
import math

import numpy as np


def norm(code):
    return str(code).replace("HP:", "").zfill(7)


def direct_parents(code, tree_lookup):
    entry = tree_lookup.get(code)
    if not entry:
        return set()
    return {norm(r[1]) for r in entry.get("parent_branches", []) if len(r) > 1 and norm(r[1]) != "-------"}


def select_terms(code_sets, tree_lookup, n_perm=200, alpha=0.05, min_count=None, low_weight=0.01, seed=1):
    """code_sets: list of sets of normalized codes, one per patient.
    Returns (weights {code: w}, report dict)."""
    n = len(code_sets)
    if min_count is None:
        min_count = max(3, math.ceil(0.03 * n))
    terms = sorted(set().union(*code_sets)) if code_sets else []
    parents = {t: direct_parents(t, tree_lookup) for t in terms}
    children = {t: set() for t in terms}
    for t in terms:
        for p in parents[t]:
            if p in children:
                children[p].add(t)
    family = {t: {t} | parents[t] | children[t] for t in terms}

    # raw terms (patients x terms) and term -> feature incidence: feature t = t, its parent or a child
    idx = {t: i for i, t in enumerate(terms)}
    R = np.zeros((n, len(terms)), dtype=bool)
    for k, cs in enumerate(code_sets):
        R[k, [idx[c] for c in cs]] = True
    F = np.zeros((len(terms), len(terms)), dtype=bool)
    for j, t in enumerate(terms):
        for c in family[t]:
            if c in idx:
                F[idx[c], j] = True
    features = lambda raw: (raw.astype(np.int32) @ F.astype(np.int32)) > 0
    B = features(R)
    counts = B.sum(0)
    test = [j for j, c in enumerate(counts) if min_count <= c <= n - min_count]
    report = dict(n_patients=n, n_terms=len(terms), n_tested=len(test), min_count=min_count,
                  n_perm=n_perm, alpha=alpha, low_weight=low_weight, selected=[], threshold=None, q={})
    if len(test) < 3:
        return {t: 1.0 for t in terms}, report
    tt = [terms[j] for j in test]
    # pairs that co-occur by construction: overlapping term sets (term, parent, children, siblings)
    fam = [family[t] for t in tt]
    sib = [parents[t] for t in tt]
    m = len(tt)
    mask = np.ones((m, m), dtype=bool)
    for i in range(m):
        mask[i, i] = False
        for j in range(i + 1, m):
            if fam[i] & fam[j] or sib[i] & sib[j]:
                mask[i, j] = mask[j, i] = False

    def q_stat(M):
        M = M.astype(float)
        sd = M.std(0)
        sd[sd == 0] = 1.0
        Z = (M - M.mean(0)) / sd
        phi = Z.T @ Z / n
        return n * np.where(mask, phi ** 2, 0.0).sum(1)

    q = q_stat(B[:, test])
    # null: each REPORTED term permuted independently across patients (keeps term frequencies and
    # the construction of the features, removes co-occurrence), then features rebuilt
    rng = np.random.default_rng(seed)
    null_max = np.array([q_stat(features(rng.permuted(R, axis=0))[:, test]).max() for _ in range(n_perm)])
    thr = float(np.quantile(null_max, 1 - alpha))
    sel = [tt[i] for i in range(m) if q[i] > thr]
    report.update(threshold=thr, q={tt[i]: float(q[i]) for i in range(m)})
    report["selected"] = sorted(sel, key=lambda t: -report["q"][t])
    if len(sel) < 2:
        report["selected"] = []
        return {t: 1.0 for t in terms}, report
    keep = set().union(*(family[t] for t in sel))
    return {t: (1.0 if t in keep else low_weight) for t in terms}, report
