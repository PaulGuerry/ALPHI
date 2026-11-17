#!/usr/local/bin/python3.14
"""
TEST 3 (Simulation 3): generates one cohort mixing patients of many diseases (by default
the 44 syndromes of Kohler et al. 2009, AJHG 85:457, Tables S2-S45), simulated as in that
article. The correct answer is one cluster per disease.

Diseases
  Symptom lists written by parse_phenomizer_si.py: code,label,freq,sex (sex = M or F for
  the SI's "Male:"/"Female:" rows, else empty). Disease name = file name without
  "symptom_list_". --disease takes the files (a shell glob is fine); every listed disease
  gets --n-per-disease patients.

Patients (Kohler et al. 2009, Material and Methods)
  - sex first, female or male 50/50; sex-specific symptoms occur only in that sex. A
    symptom is sex-specific if the SI says so (sex column) or if it lies under a
    sex-specific HPO branch (female/male genitalia, reproductive physiology, fertility)
  - each symptom drawn independently with its SI frequency
  - departure from Kohler et al.: a patient drawn with no symptom is redrawn (a case report
    always describes at least one feature; an empty patient has no distance to anyone).
    The number of redraws is reported.

Noise (--noise q, default 0)
  floor(q x number of the patient's symptoms) unrelated HPO terms are added; Kohler et al.
  used q = 0.5 ("half as many noise terms ... if the patient had nine features, we added
  four"). Noise terms come from the phenotypic-abnormality terms of the HPO (current, not
  obsolete, below an organ-system term, not an organ-system term, not sex-specific), and
  are never ancestors or descendants of a term annotated to the patient's disease, nor of
  each other. Own random stream: for a given seed the underlying patients are identical at
  every q.
  --noise-weighting uniform    (default; as Kohler et al.) every eligible term equally likely
  --noise-weighting annotated  each term weighted by the number of diseases annotated with it
                               in --hpoa (phenotype.hpoa: aspect P, NOT annotations left out,
                               old IDs mapped to current ones), so noise looks like common
                               findings; terms annotated to no disease are never drawn

Output: same format as generate_test2_patients.py (patients + evaluation_metadata). Every
patient has gene = [--label] (default TEST3); the true disease is in "disease" and
"subgroup" (1..D in the order of --disease).

Usage:
  python3.14 generate_test3_patients.py --disease phenomizer/symptom_list_*.csv \
      --n-per-disease 20 --noise 0.5 --hpo-terms hpo_terms.json --seed 1 -o test3_cohort.json
  python3.14 generate_test3_patients.py --disease phenomizer/symptom_list_*.csv --noise 0.5 \
      --noise-weighting annotated --hpoa phenotype.hpoa --seed 1
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

import generate_test2_patients as gt


def load_list(path, H):
    """[(code, label, freq, sex or None)], codes mapped to the current HPO of H."""
    rows, seen = [], set()
    for r in csv.DictReader(open(path, encoding="utf-8")):
        code = r["code"].strip()
        code = H.alt.get(code, code)
        if code not in H.terms or H.terms[code]["obsolete"]:
            print(f"WARNING {path}: {r['code']} ({r['label']}) not a current term of this HPO; left out",
                  file=sys.stderr)
            continue
        if code in seen:
            continue
        seen.add(code)
        sex = (r.get("sex") or "").strip().upper() or H.sex_of(code)
        rows.append((code, r["label"], float(r["freq"]), sex))
    if not rows:
        sys.exit(f"No symptoms read from {path}.")
    return rows


def annotation_counts(hpoa, H):
    """{HPO term: number of diseases annotated with it} from phenotype.hpoa."""
    seen = set()
    for line in open(hpoa, encoding="utf-8"):
        if line.startswith("#") or line.startswith("database_id"):
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 11 or f[10] != "P" or f[2].strip().upper() == "NOT":
            continue
        seen.add((f[0], H.alt.get(f[3], f[3])))
    counts = {}
    for _, t in seen:
        counts[t] = counts.get(t, 0) + 1
    return counts


_POOLS = {}


class NoisePool:
    """Eligible noise terms (with their weights) and the ancestor/descendant exclusions."""

    def __init__(self, H, weighting="uniform", hpoa=None):
        self.H = H
        self.pool = sorted(t for t in H.terms if H.usable(t))
        self.cum = None
        if weighting == "annotated":
            if not hpoa or not Path(hpoa).exists():
                raise ValueError("--noise-weighting annotated needs --hpoa phenotype.hpoa")
            counts = annotation_counts(hpoa, H)
            self.pool = [t for t in self.pool if counts.get(t, 0) > 0]
            self.cum = np.cumsum([counts[t] for t in self.pool], dtype=float)
        elif weighting != "uniform":
            raise ValueError(f"Unknown noise weighting {weighting!r}")

    @classmethod
    def get(cls, H, weighting, hpoa):
        key = (id(H), weighting, hpoa)
        if key not in _POOLS:
            _POOLS[key] = cls(H, weighting, hpoa)
        return _POOLS[key]

    def pick(self, rng):
        if self.cum is None:
            return self.pool[int(rng.integers(len(self.pool)))]
        return self.pool[int(np.searchsorted(self.cum, rng.random() * self.cum[-1], side="right"))]

    def related(self, a, b):
        return a == b or a in self.H.ancestors(b) or b in self.H.ancestors(a)

    def draw(self, k, disease_terms, disease_up, rng, max_tries=10000):
        """k terms unrelated to the disease's terms and to each other."""
        out, tries = [], 0
        while len(out) < k:
            tries += 1
            if tries > max_tries:
                raise RuntimeError("Could not find enough unrelated noise terms.")
            t = self.pick(rng)
            if t in disease_up or self.H.ancestors(t) & disease_terms:   # ancestor or descendant of the disease
                continue
            if any(self.related(t, u) for u in out):
                continue
            out.append(t)
        return out


def generate(disease_files, n_per_disease=20, noise=0.0, seed=None, hpo_terms="hpo_terms.json",
             label="TEST3", output_json="synthetic_patients.json", verbose=True, max_redraws=1000,
             noise_weighting="uniform", hpoa=None):
    if noise < 0:
        raise ValueError("Need noise >= 0.")
    names = [gt.disease_name(f) for f in disease_files]
    if len(set(names)) != len(names):
        raise ValueError(f"The same disease twice: {names}")
    H = gt.hpo(hpo_terms)
    lists = [load_list(f, H) for f in disease_files]
    rng = np.random.default_rng(seed)                                   # patients
    rng_n = np.random.default_rng(None if seed is None else [seed, 4])  # noise
    pool = NoisePool.get(H, noise_weighting, hpoa) if noise > 0 else None

    group = np.repeat(np.arange(len(names)), n_per_disease)
    rng.shuffle(group)
    patients, redraws = [], {n: 0 for n in names}
    for i, g in enumerate(group, start=1):
        rows = lists[g]
        f = np.array([r[2] for r in rows])
        sex = "F" if rng.random() < 0.5 else "M"
        ok = np.array([r[3] in (None, "", sex) for r in rows])
        for _ in range(max_redraws):
            draw = (rng.random(len(rows)) < f) & ok
            if draw.any():
                break
            redraws[names[g]] += 1
        else:
            raise RuntimeError(f"{names[g]}: no symptom after {max_redraws} draws.")
        syms = [rows[j][0] for j in range(len(rows)) if draw[j]]
        patients.append({"id": f"{names[g]}_{i:04d}", "gene": [label], "disease": names[g],
                         "subgroup": int(g + 1), "sex": sex, "n_true_symptoms": len(syms),
                         "symptoms": [c[3:] for c in syms]})

    n_noise = 0
    if noise > 0:
        dterms = [frozenset(r[0] for r in rows) for rows in lists]
        dup = [frozenset().union(*(H.ancestors(t) for t in d)) | d for d in dterms]
        for p in patients:
            k = int(np.floor(noise * p["n_true_symptoms"]))
            if k:
                g = p["subgroup"] - 1
                extra = pool.draw(k, dterms[g], dup[g], rng_n)
                p["noise_symptoms"] = [t[3:] for t in extra]
                p["symptoms"] = p["symptoms"] + p["noise_symptoms"]
                n_noise += k

    meta = {
        "disease_name": "phenomizer44" if len(names) == 44 else "+".join(names),
        "source_file": [str(f) for f in disease_files], "target_n_patients": len(patients),
        "effective_subgroups": len(names),
        "test3": {"diseases": names, "n_per_disease": n_per_disease, "noise": noise,
                  "noise_weighting": noise_weighting if noise > 0 else "", "seed": seed,
                  "label": label, "redraws_empty": redraws},
        "actual_cohort_sizes": {f"Subgroup {g + 1}": n_per_disease for g in range(len(names))},
        "subgroups": {},
    }
    for g, name in enumerate(names):
        members = [p for p in patients if p["subgroup"] == g + 1]
        counts = {}
        for p in members:
            for c in p["symptoms"]:
                counts[c] = counts.get(c, 0) + 1
        meta["subgroups"][str(g + 1)] = {
            "disease": name, "actual_size": len(members),
            "associated_symptoms": [{"hpo_code": c, "label": lab, "target_frequency": fr, "sex": sx or ""}
                                    for c, lab, fr, sx in lists[g]],
            "selection_frequencies": {f"HP:{c}": n / len(members) for c, n in counts.items()},
        }
    payload = {"patients": patients, "evaluation_metadata": {label: meta}}
    with open(output_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)

    if verbose:
        n_true = [p["n_true_symptoms"] for p in patients]
        n_all = [len(p["symptoms"]) for p in patients]
        print(f"{len(names)} diseases x {n_per_disease} patients = {len(patients)}; noise q = {noise:g}"
              + (f" ({noise_weighting})" if noise > 0 else ""))
        print(f"Disease symptoms per patient: mean {np.mean(n_true):.1f}, range {min(n_true)}-{max(n_true)}; "
              f"with noise: mean {np.mean(n_all):.1f} ({n_noise} noise terms)")
        r = {k: v for k, v in redraws.items() if v}
        print(f"Empty patients redrawn: {sum(r.values())}" + (f" ({r})" if r else ""))
    return payload


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disease", nargs="+", required=True, help="symptom lists from parse_phenomizer_si.py")
    ap.add_argument("--n-per-disease", type=int, default=20)
    ap.add_argument("--noise", type=float, default=0.0, help="noise terms per disease symptom (Kohler: 0.5)")
    ap.add_argument("--noise-weighting", choices=["uniform", "annotated"], default="uniform")
    ap.add_argument("--hpoa", default="phenotype.hpoa", help="for --noise-weighting annotated")
    ap.add_argument("--hpo-terms", default="hpo_terms.json")
    ap.add_argument("--label", default="TEST3", help="gene label given to every patient")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("-o", "--output-json", default="synthetic_patients.json")
    args = ap.parse_args()
    try:
        generate(sorted(args.disease), args.n_per_disease, args.noise, args.seed, args.hpo_terms,
                 args.label, args.output_json, noise_weighting=args.noise_weighting, hpoa=args.hpoa)
    except (ValueError, RuntimeError) as e:
        sys.exit(str(e))
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
