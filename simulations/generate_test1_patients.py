#!/usr/local/bin/python3.14
"""
TEST 1: generates a synthetic cohort of one artificial disease from a table written by
build_test1_table.py (one composition) or build_test1_tables.py (several compositions,
with a dispersed and a concentrated signature layout: --layout).

  D1S0  no subgroups     core + signature ranks 1..k of the sets in --null-sets, all at
                         one flat frequency (default: the D3S3 average), no structure
  D2S2  two subgroups    core + sets A, B (ranks 1..k)
  D3S3  three subgroups  core + sets A, B, C (ranks 1..k)

Frequencies
  core symptoms           core_freq from the table, in every patient
  signature symptoms      --high in their own subgroup, --low in the other subgroups
  D1S0 signature symptoms average of high and low over the subgroups of the disease
                          named by --null-like (D3S3: (high + 2 low) / 3), the same
                          symptom frequency over the cohort as in that disease, but with
                          no subgroup structure

Reporting (--reporting r, default 1)
  Imperfect reporting of the distinguishing features: each signature symptom a patient
  has is reported (kept) with probability r, so its observed frequencies become r x high
  and r x low (r x flat for D1S0). Core symptoms are always reported. The contrast
  (high/low) describes how specific a feature is; r describes how completely it is
  reported.

Signature layout (--layout, tables from build_test1_tables.py only)
  dispersed     each subgroup's signature symptoms are spread over organ systems
  concentrated  each subgroup's signature symptoms all lie in one home organ system

Term substitution (--substitution p, default 0; needs --hpo-terms)
  Imprecise or heterogeneous reporting: each reported symptom (core or signature) is
  replaced, with probability p, by a related HPO term: its parent (2/3 of cases) or a
  sibling (1/3), never an organ-system term or anything above it, never a sex-specific
  term, never another symptom of the table. Substitution uses its own random stream, so for a given seed the underlying
  patients are identical at every p. The evaluation metadata keep the true codes.

Patients are split evenly between subgroups (sizes differ by at most one). Output has
the same format as generate_synthetic_patients.py (patients + evaluation_metadata), so
pairD_phenome.py and run_gridsearch.py's evaluation read it unchanged.

Usage:
  python3.14 generate_test1_patients.py -t test1_table.csv --disease D2S2 -n 120 \
      -k 3 --high 0.8 --low 0.05 --reporting 0.75 --seed 1 -o synthetic_patients.json
  python3.14 generate_test1_patients.py -t test1_table_C1.csv --layout concentrated --disease D3S3 \
      -n 90 -k 6 --substitution 0.25 --hpo-terms hpo_terms.json --seed 1
"""
import argparse
import csv
import json
import sys

import numpy as np

DESIGNS = {"D1S0": 1, "D2S2": 2, "D3S3": 3}
SETS = "ABC"


def load_table(path, layout=None):
    """Core rows and signature sets of one layout ('dispersed' by default when the table has
    a 'layout' column; tables from build_test1_table.py have one layout only)."""
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if "layout" in (rows[0] if rows else {}):
        layout = layout or "dispersed"
        rows = [r for r in rows if r["role"] == "core" or r["layout"] == layout]
        if not any(r["role"] == "signature" for r in rows):
            sys.exit(f"Table error: no signature rows with layout '{layout}' in {path}.")
    elif layout not in (None, "dispersed"):
        sys.exit(f"{path} has no 'layout' column: only the dispersed layout is available.")
    core = [r for r in rows if r["role"] == "core"]
    sig = {st: sorted((r for r in rows if r["role"] == "signature" and r["set"] == st),
                      key=lambda r: int(r["rank"])) for st in SETS}
    codes = [r["hpo"] for r in rows]
    if len(codes) != len(set(codes)):
        sys.exit("Table error: a symptom appears twice.")
    for r in core:
        f = float(r["core_freq"])
        if not 0 <= f <= 1:
            sys.exit(f"Table error: core frequency {f} out of range for {r['hpo']}.")
    k_max = min(len(v) for v in sig.values())
    for rk in range(k_max):
        systems = {sig[st][rk]["organ_system"] for st in SETS}
        if len(systems) != 1 and layout != "concentrated":
            print(f"WARNING: rank {rk + 1} signature symptoms are in different organ systems: {systems}",
                  file=sys.stderr)
    return core, sig, k_max


# ---------------------------------------------------------------------------------------
# Term substitution
# ---------------------------------------------------------------------------------------
ORGAN_SYSTEM_CODES = {"HP:" + c for c in (
    "0033127", "0040064", "0000707", "0001939", "0000119", "0025354", "0002715", "0000478",
    "0001574", "0001608", "0025031", "0000598", "0001626", "0001197", "0002664", "0000152",
    "0000818", "0025142", "0001507", "0045027", "0002086", "0001871", "0000769")}
TOO_GENERAL = ORGAN_SYSTEM_CODES | {"HP:0000118", "HP:0000001"}
SEX_SPECIFIC = {"HP:0010461", "HP:0010460", "HP:0012874", "HP:0030012", "HP:0000769",
                "HP:0012041", "HP:0000868"}
_SUBST = {}


class TermSubstituter:
    """Replaces an HPO term by its parent (p_parent) or a sibling, within its organ system."""

    def __init__(self, hpo_terms_path, p_parent=2 / 3):
        d = json.load(open(hpo_terms_path, encoding="utf-8"))
        self.terms = d["terms"]
        self.p_parent = p_parent
        self._anc = {}
        self._cand = {}

    def ancestors(self, t):
        hit = self._anc.get(t)
        if hit is None:
            hit = set()
            for p in self.terms.get(t, {}).get("parents", []):
                hit |= {p} | self.ancestors(p)
            self._anc[t] = hit
        return hit

    def usable(self, t):
        v = self.terms.get(t)
        return (v is not None and not v["obsolete"] and t not in TOO_GENERAL
                and not (self.ancestors(t) | {t}) & SEX_SPECIFIC
                and self.ancestors(t) & ORGAN_SYSTEM_CODES)

    def candidates(self, t):
        hit = self._cand.get(t)
        if hit is None:
            parents = [p for p in self.terms.get(t, {}).get("parents", []) if self.usable(p)]
            sibs = sorted({c for p in self.terms.get(t, {}).get("parents", [])
                           for c in self.terms.get(p, {}).get("children", [])
                           if c != t and self.usable(c)})
            hit = self._cand[t] = (parents, sibs)
        return hit

    def substitute(self, t, rng, avoid=frozenset()):
        """avoid: codes of the table, never used as replacements (a replacement must not
        turn into another symptom of the design)."""
        parents, sibs = self.candidates(t)
        parents = [x for x in parents if x not in avoid]
        sibs = [x for x in sibs if x not in avoid]
        use_parent = parents and (not sibs or rng.random() < self.p_parent)
        pool = parents if use_parent else sibs
        return pool[int(rng.integers(len(pool)))] if pool else t


def substituter(path):
    if path not in _SUBST:
        _SUBST[path] = TermSubstituter(path)
    return _SUBST[path]


def build_design(core, sig, disease, k, high, low, null_like, null_sets, reporting=1.0):
    """Returns (symptoms, prob matrix [n_subgroups x n_symptoms], subgroup -> signature symptoms)."""
    n_sub = DESIGNS[disease]
    symptoms = [(r["hpo"], r["label"], "core", "") for r in core]
    probs = [[float(r["core_freq"])] * n_sub for r in core]
    signatures = {s: [] for s in range(1, n_sub + 1)}
    if n_sub == 1:
        m = DESIGNS[null_like]
        flat = (high + (m - 1) * low) / m
        for st in null_sets:
            for r in sig[st][:k]:
                symptoms.append((r["hpo"], r["label"], "signature", st))
                probs.append([flat * reporting])
    else:
        for s, st in enumerate(SETS[:n_sub], start=1):
            for r in sig[st][:k]:
                symptoms.append((r["hpo"], r["label"], "signature", st))
                probs.append([(high if t == s else low) * reporting for t in range(1, n_sub + 1)])
                signatures[s].append((r["hpo"], r["label"], high * reporting))
    return symptoms, np.array(probs).T, signatures


def generate(table, disease, n_patients, sig_size, high=0.8, low=0.05, null_like="D3S3",
             null_sets=None, seed=None, output_json="synthetic_patients.json", verbose=True,
             reporting=1.0, layout=None, substitution=0.0, hpo_terms=None):
    """Generates one cohort and writes it to output_json. Returns the payload."""
    core, sig, k_max = load_table(table, layout)
    if not 0 <= substitution <= 1:
        raise ValueError("Need 0 <= substitution <= 1.")
    if substitution > 0 and not hpo_terms:
        raise ValueError("Term substitution needs hpo_terms (path to hpo_terms.json).")
    if not 0 <= sig_size <= k_max:
        raise ValueError(f"sig_size must be between 0 and {k_max} (signature ranks in the table).")
    if not 0 <= low <= high <= 1:
        raise ValueError("Need 0 <= low <= high <= 1.")
    if not 0 < reporting <= 1:
        raise ValueError("Need 0 < reporting <= 1.")
    null_sets = null_sets or SETS[:DESIGNS[null_like]]

    symptoms, P, signatures = build_design(core, sig, disease, sig_size, high, low, null_like, null_sets,
                                           reporting)
    n_sub = P.shape[0]
    rng = np.random.default_rng(seed)

    # even split, in random order
    sub_of = np.array([i % n_sub for i in range(n_patients)])
    rng.shuffle(sub_of)
    patients = []
    codes = [s[0] for s in symptoms]
    for i, s in enumerate(sub_of, start=1):
        mask = rng.random(len(codes)) < P[s]
        patients.append({"id": f"{disease}_sub{s + 1}_{i:03d}", "gene": [disease],
                         "subgroup": int(s + 1),
                         "symptoms": [c.replace("HP:", "") for c, m in zip(codes, mask) if m]})

    # term substitution, on its own random stream (base patients identical at every p)
    n_subst = 0
    if substitution > 0:
        sub = substituter(hpo_terms)
        rng_s = np.random.default_rng(None if seed is None else [seed, 1])
        with open(table, encoding="utf-8") as f:
            avoid = frozenset(r["hpo"] for r in csv.DictReader(f))
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

    # metadata in the format of generate_synthetic_patients.py
    sizes = {s: int((sub_of == s - 1).sum()) for s in range(1, n_sub + 1)}
    meta = {
        "disease_name": disease, "source_file": str(table), "target_n_patients": n_patients,
        "effective_subgroups": n_sub, "n_subgroup_signature_symptoms": sig_size,
        "test1": {"high": high, "low": low, "reporting": reporting, "seed": seed,
                  "layout": layout or "dispersed", "substitution": substitution,
                  "null_like": null_like if n_sub == 1 else None,
                  "null_sets": null_sets if n_sub == 1 else None},
        "disease_specific_symptoms": [{"hpo_code": r["hpo"], "label": r["label"],
                                       "target_frequency": float(r["core_freq"])} for r in core],
        "actual_cohort_sizes": {f"Subgroup {s}": sizes[s] for s in sizes},
        "subgroups": {},
    }
    for s in range(1, n_sub + 1):
        members = [p for p in patients if p["subgroup"] == s]
        counts = {}
        for p in members:
            for c in p["symptoms"]:
                counts[c] = counts.get(c, 0) + 1
        if n_sub == 1:   # as before: with no subgroups, every symptom is "associated"
            assoc = [{"hpo_code": c, "label": lab, "target_frequency": float(P[0][j])}
                     for j, (c, lab, _, _) in enumerate(symptoms)]
        else:
            assoc = [{"hpo_code": c, "label": lab, "target_frequency": f} for c, lab, f in signatures[s]]
        meta["subgroups"][str(s)] = {
            "actual_size": sizes[s], "associated_symptoms": assoc,
            "selection_frequencies": {f"HP:{c}": n / len(members) for c, n in counts.items()} if members else {},
        }

    payload = {"patients": patients, "evaluation_metadata": {disease: meta}}
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)

    if verbose:
        n_sym = [len(p["symptoms"]) for p in patients]
        print(f"{disease}: {n_patients} patients, subgroups {sizes}, k = {sig_size}, high {high}, low {low}, "
              f"reporting {reporting}, layout {layout or 'dispersed'}, substitution {substitution}"
              + (f", flat {P[0][len(core)]:.3f} ({null_sets})" if n_sub == 1 and sig_size else ""))
        print(f"Symptoms per patient: mean {np.mean(n_sym):.1f}, range {min(n_sym)}-{max(n_sym)}"
              + (f"; {n_subst} terms substituted" if substitution > 0 else ""))
    return payload


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-t", "--table", default="test1_table.csv")
    ap.add_argument("--disease", choices=list(DESIGNS), required=True)
    ap.add_argument("-n", "--n-patients", type=int, required=True)
    ap.add_argument("-k", "--sig-size", type=int, required=True, help="Signature symptoms per subgroup.")
    ap.add_argument("--high", type=float, default=0.8, help="Signature frequency in its own subgroup.")
    ap.add_argument("--low", type=float, default=0.05, help="Signature frequency in the other subgroups.")
    ap.add_argument("--reporting", type=float, default=1.0,
                    help="Probability that a signature symptom a patient has is reported (default 1).")
    ap.add_argument("--layout", choices=["dispersed", "concentrated"], default=None,
                    help="Signature layout (tables from build_test1_tables.py).")
    ap.add_argument("--substitution", type=float, default=0.0,
                    help="Probability that a reported term is replaced by a related term.")
    ap.add_argument("--hpo-terms", default="hpo_terms.json", help="For --substitution.")
    ap.add_argument("--null-like", choices=["D2S2", "D3S3"], default="D3S3",
                    help="D1S0 only: which disease's average signature frequency to use.")
    ap.add_argument("--null-sets", default=None,
                    help="D1S0 only: signature sets included (default: those of --null-like, e.g. ABC).")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("-o", "--output-json", default="synthetic_patients.json")
    args = ap.parse_args()
    try:
        generate(args.table, args.disease, args.n_patients, args.sig_size, args.high, args.low,
                 args.null_like, args.null_sets, args.seed, args.output_json, reporting=args.reporting,
                 layout=args.layout, substitution=args.substitution, hpo_terms=args.hpo_terms)
    except ValueError as e:
        sys.exit(str(e))
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
