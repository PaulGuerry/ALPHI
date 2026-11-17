#!/usr/local/bin/python3.14
"""
Estimates how heterogeneously a feature is coded within groups of patients, as a
counterpart of the term-substitution probability p of the simulations.

Within each group (an ALPHI cluster, or a true subgroup of a simulated cohort), the terms
are taken in order of decreasing frequency; each term not yet assigned opens a "feature
family" made of itself (the modal term) and the unassigned terms one step away from it in
the HPO: its parents, its children and its siblings (the substitutions of the simulations:
parent or sibling). For every patient reporting at least one term of a family, the feature
counts as coded with a substitute when the patient does not report the modal term.

  p_obs = patients x families coded with a substitute / patients x families reported

At p = 0, p_obs is not 0 (distinct features can be HPO siblings), so p_obs is calibrated
on simulated cohorts generated at known p (--calibrate) and the real cohort is read off
that curve.

Usage:
  # real cohort: groups = clusters of an ALPHI run
  python3.14 estimate_substitution.py --hpo-terms hpo_terms.json \
      --patients 261013_patientData.json --clusters clusters.json
  # simulated cohorts at known p: groups = true subgroups
  python3.14 estimate_substitution.py --hpo-terms hpo_terms.json --synthetic sim_p0.json sim_p0.25.json ...
"""
import argparse
import json
from collections import Counter


class HPO:
    def __init__(self, path):
        d = json.load(open(path, encoding="utf-8"))
        self.t = d["terms"]
        self.alt = d.get("alt_id_map", {})

    def norm(self, code):
        c = code if str(code).startswith("HP:") else "HP:" + str(code).zfill(7)
        return self.alt.get(c, c)

    def parents(self, c):
        return set(self.t.get(c, {}).get("parents", []))

    def related(self, a, b):
        pa, pb = self.parents(a), self.parents(b)
        return a in pb or b in pa or bool(pa & pb)


def p_obs(groups, H):
    """groups: {name: [set of terms per patient]} -> {name: (substituted, reported, n_families)}"""
    res = {}
    for g, pats in groups.items():
        freq = Counter(t for p in pats for t in p)
        order = sorted(freq, key=lambda t: (-freq[t], t))
        assigned, fams = set(), []
        for t in order:
            if t in assigned:
                continue
            fam = {t} | {u for u in order if u not in assigned and u != t and H.related(u, t)}
            assigned |= fam
            fams.append((t, fam))
        sub = rep = 0
        for modal, fam in fams:
            for p in pats:
                if p & fam:
                    rep += 1
                    sub += modal not in p
        res[g] = (sub, rep, sum(len(f) > 1 for _, f in fams))
    return res


def show(label, res):
    S = sum(v[0] for v in res.values())
    R = sum(v[1] for v in res.values())
    parts = ", ".join(f"{g}: {s / r:.3f} ({s}/{r})" for g, (s, r, _) in res.items())
    print(f"{label}: p_obs = {S / R:.3f}   [{parts}]")
    return S / R


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hpo-terms", required=True)
    ap.add_argument("--patients", help="patient JSON (PytheasDB format)")
    ap.add_argument("--clusters", help="clusters.json of an ALPHI run on those patients")
    ap.add_argument("--solution", default="optimal")
    ap.add_argument("--synthetic", nargs="*", default=[], help="simulated cohorts (groups = true subgroups)")
    a = ap.parse_args()
    H = HPO(a.hpo_terms)

    for f in a.synthetic:
        d = json.load(open(f, encoding="utf-8"))
        groups = {}
        for p in d["patients"]:
            groups.setdefault(str(p.get("subgroup")), []).append({H.norm(c) for c in p["symptoms"]})
        show(f, p_obs(groups, H))

    if a.patients and a.clusters:
        db = {str(p["id"]): p for p in json.load(open(a.patients, encoding="utf-8"))}
        sol = json.load(open(a.clusters, encoding="utf-8"))["clustering_results"][a.solution]
        groups = {}
        for c in sol["clusters"]:
            if c.get("cluster_id") == -1:
                continue
            name = str(c.get("cluster_name") or c.get("cluster_id"))
            for p in c.get("patients") or []:
                pid = str(p["id"]) if isinstance(p, dict) else str(p)
                syms = db[pid].get("symptoms") or []
                groups.setdefault(name, []).append({H.norm(s[0] if isinstance(s, list) else s) for s in syms})
        show("clusters", p_obs(groups, H))
        show("whole cohort", p_obs({"all": [p for v in groups.values() for p in v]}, H))


if __name__ == "__main__":
    main()
