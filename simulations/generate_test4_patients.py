#!/usr/local/bin/python3.14
"""
Simulation 4: a cohort with a combined subgroup and a marker with two causes, built on the
symptom tables of Simulation 2 (build_test1_tables.py) and on generate_test1_patients.py.

Modelled on the MYO5B cohort:
  subgroup 1, A   (MVID-like)       core symptoms + signature set A
  subgroup 2, B   (PFIC-like)       core symptoms + signature set B
  subgroup 3, AB  (combined)        core symptoms + sets A and B, + --ab-specific symptoms of set C
                                    present in this subgroup only (as fibrosis and cirrhosis
                                    in the MVID+PFIC cluster)
Signature symptoms are present at --high in their subgroup(s) and --low elsewhere, times the
reporting rate r, as in Simulation 2.
Marker with two causes: in subgroup A, the --secondary symptoms of set B (default: rank 4,
the digestive symptom of the set) are present with probability --f instead of --low, without
the rest of set B (as isolated cholestasis in MVID with parenteral nutrition). The patients
of A who carry one of them are flagged "secondary" (ground truth: subgroup A).
Partial expression in AB (--ab-b): B's signature symptoms are present in AB at --high times
--ab-b (1 = as in B), as combined patients described mainly by their intestinal disease with
one or two liver terms.
Simulation 5 uses this generator with a table of meaningful terms (test5_table_myo5b.csv:
intestinal set A, cholestatic set B whose rank 1 is Cholestasis, fibrotic set C) and the
marker with two causes at rank 1 (--secondary 1): the term that names B is also present,
alone, in some patients of A.
Subgroup sizes are fixed (--sizes A B AB, default 70 45 25, as the MYO5B clusters 69/43/25),
assigned in random order. Term substitution as in Simulation 2 (--substitution p).

Usage:
  python3.14 generate_test4_patients.py -t test1_table_C1.csv --hpo-terms hpo_terms.json \
      --sizes 70 45 25 -k 6 --f 0.3 --ab-specific 3 --substitution 0.1 --seed 1 -o cohort.json
"""
import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("generate_test1_patients", _here / "generate_test1_patients.py")
g1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g1)

LABEL = "D4AB"
NAMES = {1: "A", 2: "B", 3: "AB"}


def generate(table, sizes=(70, 45, 25), sig_size=6, high=0.8, low=0.05, f=0.3, secondary=(4,),
             ab_specific=3, reporting=1.0, layout="dispersed", substitution=0.0, hpo_terms=None,
             seed=None, output_json="synthetic_patients.json", verbose=True, ab_b=1.0):
    core, sig, k_max = g1.load_table(table, layout)
    if not 1 <= sig_size <= k_max or not 0 <= ab_specific <= k_max:
        raise ValueError(f"sig_size and ab_specific must be at most {k_max}.")
    if any(not 1 <= r <= sig_size for r in secondary):
        raise ValueError("--secondary ranks must be signature ranks of set B (1..k).")
    if substitution > 0 and not hpo_terms:
        raise ValueError("Term substitution needs hpo_terms.")
    sec_codes = {sig["B"][r - 1]["hpo"] for r in secondary}

    # symptoms and presence probabilities per subgroup (columns: A, B, AB)
    symptoms, probs = [], []
    for r in core:
        symptoms.append((r["hpo"], r["label"], "core", ""))
        probs.append([float(r["core_freq"])] * 3)
    for st, in_groups in (("A", (0, 2)), ("B", (1, 2))):
        for r in sig[st][:sig_size]:
            p = [(high if g in in_groups else low) * reporting for g in range(3)]
            if st == "B":
                p[2] = high * ab_b * reporting            # B's symptoms in AB (partial expression)
            if st == "B" and r["hpo"] in sec_codes:
                p[0] = f * reporting                      # the marker with two causes, in A
            symptoms.append((r["hpo"], r["label"], "signature", st))
            probs.append(p)
    for r in sig["C"][:ab_specific]:
        symptoms.append((r["hpo"], r["label"], "ab_specific", "C"))
        probs.append([low * reporting, low * reporting, high * reporting])
    P = np.array(probs).T                                 # 3 x n_symptoms

    rng = np.random.default_rng(seed)
    sub_of = np.repeat([0, 1, 2], sizes)
    rng.shuffle(sub_of)
    codes = [s[0] for s in symptoms]
    patients = []
    for i, s in enumerate(sub_of, start=1):
        mask = rng.random(len(codes)) < P[s]
        present = [c for c, m in zip(codes, mask) if m]
        patients.append({"id": f"{LABEL}_{NAMES[s + 1]}_{i:03d}", "gene": [LABEL], "subgroup": int(s + 1),
                         "group_name": NAMES[s + 1],
                         "secondary": int(s == 0 and any(c in sec_codes for c in present)),
                         "symptoms": [c.replace("HP:", "") for c in present]})

    n_subst = 0
    if substitution > 0:                                 # own random stream, as in Simulation 2
        sub = g1.substituter(hpo_terms)
        rng_s = np.random.default_rng(None if seed is None else [seed, 1])
        with open(table, encoding="utf-8") as fh:
            avoid = frozenset(r["hpo"] for r in csv.DictReader(fh))
        for p in patients:
            new = []
            for c in p["symptoms"]:
                code = "HP:" + c
                if rng_s.random() < substitution:
                    rep = sub.substitute(code, rng_s, avoid)
                    n_subst += rep != code
                    code = rep
                if code[3:] not in new:
                    new.append(code[3:])
            p["symptoms"] = new

    meta = {"disease_name": LABEL, "source_file": str(table), "sizes": dict(zip("A B AB".split(), map(int, sizes))),
            "test4": dict(high=high, low=low, f=f, secondary=[sig["B"][r - 1]["label"] for r in secondary],
                          ab_specific=[r["label"] for r in sig["C"][:ab_specific]], reporting=reporting,
                          layout=layout, substitution=substitution, seed=seed, sig_size=sig_size, ab_b=ab_b),
            "n_secondary": sum(p["secondary"] for p in patients)}
    with open(output_json, "w", encoding="utf-8") as fh:
        json.dump({"patients": patients, "evaluation_metadata": {LABEL: meta}}, fh, indent=1)
    if verbose:
        print(f"{LABEL}: sizes {meta['sizes']}, k = {sig_size}, f = {f} ({meta['n_secondary']} A patients with the "
              f"secondary marker), AB-specific {ab_specific}, r {reporting}, p {substitution}"
              + (f"; {n_subst} terms substituted" if substitution > 0 else ""))
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-t", "--table", required=True)
    ap.add_argument("--hpo-terms", default="hpo_terms.json")
    ap.add_argument("--sizes", nargs=3, type=int, default=[70, 45, 25], metavar=("A", "B", "AB"))
    ap.add_argument("-k", "--sig-size", type=int, default=6)
    ap.add_argument("--high", type=float, default=0.8)
    ap.add_argument("--low", type=float, default=0.05)
    ap.add_argument("--f", type=float, default=0.3, help="frequency of the secondary marker in subgroup A")
    ap.add_argument("--secondary", nargs="+", type=int, default=[4], help="ranks of set B used as secondary markers")
    ap.add_argument("--ab-specific", type=int, default=3, help="symptoms of set C present in AB only")
    ap.add_argument("--ab-b", type=float, default=1.0,
                    help="presence of B's signature symptoms in AB relative to B (1 = as in B; 0.4 = partial)")
    ap.add_argument("--reporting", type=float, default=1.0)
    ap.add_argument("--layout", default="dispersed", choices=["dispersed", "concentrated"])
    ap.add_argument("--substitution", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("-o", "--output-json", default="synthetic_patients.json")
    a = ap.parse_args()
    generate(a.table, a.sizes, a.sig_size, a.high, a.low, a.f, a.secondary, a.ab_specific, a.reporting, a.layout,
             a.substitution, a.hpo_terms, a.seed, a.output_json, ab_b=a.ab_b)


if __name__ == "__main__":
    main()
