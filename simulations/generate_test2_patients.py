#!/usr/local/bin/python3.14
"""
TEST 2: generates a synthetic cohort mixing patients of one, two or three real diseases,
each simulated from its Orphanet symptom list.

Diseases
  Each disease is read from its Orphanet symptom list:
    .txt  the list as copied from the Orphanet page (frequency headers followed by
          "Label HP:0000000" lines), read with parse_orphanet_symptoms.py's rules
    .csv  output of parse_orphanet_symptoms.py --with-class ("Orphanet Class" column);
          a CSV without that column has fixed frequencies and cannot be sampled
  The disease name is the file name without "symptom_list_" (e.g. symptom_list_wilson.txt
  -> wilson).

Frequencies (--frequency, default sample)
  sample    for every cohort, each symptom's frequency is drawn uniformly within its
            Orphanet class (e.g. Frequent = 30-79 %), once per disease and cohort: the
            Orphanet classes are ranges, so the exact frequency in a real cohort is unknown.
            Drawn on its own random stream, so for a given seed it does not depend on the
            substitution level.
  midpoint  the middle of each class
  fixed     the Frequency column of a CSV (as written by parse_orphanet_symptoms.py)
  Always present / obligate = 100 %; excluded symptoms are left out.

Group sizes (--proportions, e.g. 3:2:1, in the order of --disease)
  Patients are split between the diseases in these proportions (largest-remainder
  rounding, every disease at least 2 patients), in random order. One disease = a
  single-disease control (correct answer: one cluster).

Sex
  Every patient is female or male (50/50). Symptoms under a sex-specific branch (female
  or male genitalia, reproductive physiology, fertility) occur only in patients of that
  sex, at their Orphanet frequency; e.g. a menstrual-cycle symptom never occurs in males.

Reporting (--reporting r, default 1)
  Incomplete reporting: each symptom a patient has is reported with probability r
  (all symptoms, not only the distinguishing ones as in TEST 1). Orphanet lists are
  long, so a fully reported simulated patient has far more symptoms than a typical
  literature case report; r < 1 brings the numbers closer. Own random stream: for a given
  seed the patients' true symptoms do not depend on r.

Term substitution (--substitution p, default 0; needs --hpo-terms)
  As in TEST 1: each reported symptom is replaced, with probability p, by its parent
  (2/3 of cases) or a sibling (1/3), never an organ-system term or anything above it, never
  a sex-specific term, never a symptom of one of the diseases in the cohort. Own random
  stream: for a given seed the underlying patients are identical at every p.

Output: same format as generate_test1_patients.py (patients + evaluation_metadata). Every
patient has gene = [--label] (default TEST2), so pairD_phenome.py treats the mixture as one
cohort and never sees the true disease; the true disease is in "disease" and "subgroup"
(1, 2, 3 in the order of --disease).

Usage:
  python3.14 generate_test2_patients.py --disease symptom_list_wilson.txt symptom_list_alagille.txt \
      --proportions 2:1 -n 90 --substitution 0.25 --hpo-terms hpo_terms.json --seed 1
  python3.14 generate_test2_patients.py --disease symptom_list_marfan.txt -n 60 --seed 1   # control
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------------------
# Orphanet classes (same ranges as parse_orphanet_symptoms.py)
# ---------------------------------------------------------------------------------------
FREQ_RANGES = {
    "always present": (1.00, 1.00),
    "obligate": (1.00, 1.00),
    "very frequent": (0.80, 0.99),
    "frequent": (0.30, 0.79),
    "occasional": (0.05, 0.29),
    "very rare": (0.01, 0.04),
    "rare": (0.01, 0.04),
    "excluded": None,
}
SYMPTOM = re.compile(r'^(.*?)\s+(HP:\d{7})$')

# sex-specific branches: symptoms below them occur only in that sex
FEMALE_ONLY = {"HP:0010460", "HP:0030012", "HP:0000868"}   # female genitalia, female reproductive physiology, female fertility
MALE_ONLY = {"HP:0010461", "HP:0012874", "HP:0012041"}     # male genitalia, male reproductive physiology, male fertility

# term substitution: never into these
ORGAN_SYSTEM_CODES = {"HP:" + c for c in (
    "0033127", "0040064", "0000707", "0001939", "0000119", "0025354", "0002715", "0000478",
    "0001574", "0001608", "0025031", "0000598", "0001626", "0001197", "0002664", "0000152",
    "0000818", "0025142", "0001507", "0045027", "0002086", "0001871", "0000769")}
TOO_GENERAL = ORGAN_SYSTEM_CODES | {"HP:0000118", "HP:0000001"}
SEX_SPECIFIC = FEMALE_ONLY | MALE_ONLY | {"HP:0000769"}


def _header_class(line):
    key = re.sub(r'\s*\(.*\)\s*$', '', line).strip().lower()
    return key if key in FREQ_RANGES else None


def disease_name(path):
    return Path(path).stem.replace("symptom_list_", "")


def load_disease(path):
    """[(code, label, class or None, fixed frequency or None)] for one disease."""
    path = Path(path)
    rows, seen = [], set()
    if path.suffix.lower() == ".txt":
        current = None
        for n, raw in enumerate(open(path, encoding="utf-8"), 1):
            line = raw.strip()
            if not line:
                continue
            cls = _header_class(line)
            if cls is not None:
                current = cls
                continue
            m = SYMPTOM.match(line)
            if not m or current is None:
                print(f"WARNING {path}:{n}: line skipped: {line!r}", file=sys.stderr)
                continue
            label, code = m.group(1).strip(), m.group(2)
            if code in seen or FREQ_RANGES[current] is None:
                continue
            seen.add(code)
            rows.append((code, label, current, None))
    else:
        for r in csv.DictReader(open(path, encoding="utf-8")):
            code = r["HPO Code"].strip()
            if code in seen:
                continue
            seen.add(code)
            cls = (r.get("Orphanet Class") or "").strip().lower() or None
            if cls is not None and cls not in FREQ_RANGES:
                sys.exit(f"{path}: unknown Orphanet class {cls!r} for {code}.")
            if cls is not None and FREQ_RANGES[cls] is None:
                continue
            rows.append((code, r["HPO Label or Descriptor"], cls, float(r["Frequency"])))
    if not rows:
        sys.exit(f"No symptoms read from {path}.")
    return rows


def frequencies(rows, mode, rng):
    """One frequency per symptom."""
    out = []
    for code, label, cls, fixed in rows:
        if mode == "fixed" or cls is None:
            if fixed is None:
                raise ValueError(f"No frequency for {code} ({label}).")
            if mode != "fixed" and not getattr(frequencies, "_warned", False):
                print("WARNING: CSV without an 'Orphanet Class' column: frequencies fixed, not sampled "
                      "(write it with parse_orphanet_symptoms.py --with-class).", file=sys.stderr)
                frequencies._warned = True
            out.append(fixed)
            continue
        lo, hi = FREQ_RANGES[cls]
        out.append(lo if lo == hi else (lo + hi) / 2 if mode == "midpoint" else float(rng.uniform(lo, hi)))
    return out


def split_sizes(n, proportions):
    """Largest-remainder rounding of n x proportions, every group at least 2."""
    w = np.array(proportions, dtype=float)
    w = w / w.sum()
    if n < 2 * len(w):
        raise ValueError(f"N = {n} too small for {len(w)} groups of at least 2 patients.")
    raw = n * w
    sizes = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - sizes), kind="stable")[: n - sizes.sum()]:
        sizes[i] += 1
    while (sizes < 2).any():                        # take from the largest group
        sizes[np.argmin(sizes)] += 1
        sizes[np.argmax(sizes)] -= 1
    return [int(x) for x in sizes]


# ---------------------------------------------------------------------------------------
# HPO helpers: ancestors, sex-specific terms, term substitution
# ---------------------------------------------------------------------------------------
_HPO = {}


class HPO:
    def __init__(self, path, p_parent=2 / 3):
        d = json.load(open(path, encoding="utf-8"))
        self.terms = d["terms"]
        self.alt = d.get("alt_id_map", {})
        self.p_parent = p_parent
        self._anc, self._cand = {}, {}

    def ancestors(self, t):
        hit = self._anc.get(t)
        if hit is None:
            hit = set()
            for p in self.terms.get(t, {}).get("parents", []):
                hit |= {p} | self.ancestors(p)
            self._anc[t] = hit
        return hit

    def sex_of(self, t):
        """'F' or 'M' for a sex-specific term, else None."""
        up = self.ancestors(self.alt.get(t, t)) | {t}
        if up & FEMALE_ONLY:
            return "F"
        if up & MALE_ONLY:
            return "M"
        return None

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
        parents, sibs = self.candidates(t)
        parents = [x for x in parents if x not in avoid]
        sibs = [x for x in sibs if x not in avoid]
        use_parent = parents and (not sibs or rng.random() < self.p_parent)
        pool = parents if use_parent else sibs
        return pool[int(rng.integers(len(pool)))] if pool else t


def hpo(path):
    if path not in _HPO:
        _HPO[path] = HPO(path)
    return _HPO[path]


# ---------------------------------------------------------------------------------------
# Cohort
# ---------------------------------------------------------------------------------------

def generate(disease_files, n_patients, proportions=None, seed=None, substitution=0.0,
             hpo_terms="hpo_terms.json", frequency="sample", label="TEST2",
             output_json="synthetic_patients.json", verbose=True, reporting=1.0):
    """Generates one cohort and writes it to output_json. Returns the payload."""
    names = [disease_name(f) for f in disease_files]
    if len(set(names)) != len(names):
        raise ValueError(f"The same disease twice: {names}")
    proportions = proportions or [1] * len(names)
    if len(proportions) != len(names):
        raise ValueError("One proportion per disease is needed.")
    if not 0 <= substitution <= 1:
        raise ValueError("Need 0 <= substitution <= 1.")
    if not 0 < reporting <= 1:
        raise ValueError("Need 0 < reporting <= 1.")
    H = hpo(hpo_terms)                     # needed for the sex-specific symptoms in any case

    lists = [load_disease(f) for f in disease_files]
    rng = np.random.default_rng(seed)                                   # patients
    rng_f = np.random.default_rng(None if seed is None else [seed, 2])  # frequencies
    rng_s = np.random.default_rng(None if seed is None else [seed, 1])  # substitution
    rng_r = np.random.default_rng(None if seed is None else [seed, 3])  # reporting
    freqs = [frequencies(rows, frequency, rng_f) for rows in lists]
    sexes_of = [[H.sex_of(code) for code, *_ in rows] for rows in lists]

    sizes = split_sizes(n_patients, proportions)
    group = np.repeat(np.arange(len(names)), sizes)
    rng.shuffle(group)
    patients = []
    for i, g in enumerate(group, start=1):
        sex = "F" if rng.random() < 0.5 else "M"
        rows, f, sx = lists[g], np.array(freqs[g]), sexes_of[g]
        draw = rng.random(len(rows)) < f                     # true symptoms
        told = rng_r.random(len(rows)) < reporting           # reported (always drawn: paired across r)
        syms = [rows[j][0][3:] for j in range(len(rows)) if draw[j] and told[j] and sx[j] in (None, sex)]
        patients.append({"id": f"{names[g]}_{i:03d}", "gene": [label], "disease": names[g],
                         "subgroup": int(g + 1), "sex": sex, "symptoms": syms})

    n_subst = 0
    if substitution > 0:
        avoid = frozenset(code for rows in lists for code, *_ in rows)
        for p in patients:
            new = []
            for c in p["symptoms"]:
                code = "HP:" + c
                if rng_s.random() < substitution:
                    rep = H.substitute(code, rng_s, avoid)
                    n_subst += rep != code
                    code = rep
                if code[3:] not in new:
                    new.append(code[3:])
            p["symptoms"] = new

    # metadata (format of generate_test1_patients.py)
    scen = "+".join(names)
    meta = {
        "disease_name": scen, "source_file": [str(f) for f in disease_files], "target_n_patients": n_patients,
        "effective_subgroups": len(names),
        "test2": {"diseases": names, "proportions": list(proportions), "seed": seed, "frequency": frequency,
                  "substitution": substitution, "reporting": reporting, "label": label},
        "actual_cohort_sizes": {f"Subgroup {g + 1}": sizes[g] for g in range(len(names))},
        "subgroups": {},
    }
    for g, name in enumerate(names):
        members = [p for p in patients if p["subgroup"] == g + 1]
        counts = {}
        for p in members:
            for c in p["symptoms"]:
                counts[c] = counts.get(c, 0) + 1
        meta["subgroups"][str(g + 1)] = {
            "disease": name, "actual_size": sizes[g],
            "associated_symptoms": [{"hpo_code": code, "label": lab, "target_frequency": round(fr, 4)}
                                    for (code, lab, _, _), fr in zip(lists[g], freqs[g])],
            "selection_frequencies": {f"HP:{c}": n / len(members) for c, n in counts.items()},
        }
    payload = {"patients": patients, "evaluation_metadata": {label: meta}}
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)

    if verbose:
        n_sym = [len(p["symptoms"]) for p in patients]
        print(f"{scen}: {n_patients} patients, sizes {dict(zip(names, sizes))}, frequencies {frequency}, "
              f"reporting {reporting}, substitution {substitution}")
        print(f"Symptoms per patient: mean {np.mean(n_sym):.1f}, range {min(n_sym)}-{max(n_sym)}"
              + (f"; {n_subst} terms substituted" if substitution > 0 else ""))
    return payload


def parse_proportions(text):
    try:
        return [float(x) for x in text.split(":")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"proportions like 3:2:1, not {text!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disease", nargs="+", required=True, help="1-3 Orphanet symptom lists (.txt or .csv).")
    ap.add_argument("--proportions", type=parse_proportions, default=None, help="e.g. 2:1 or 3:2:1")
    ap.add_argument("-n", "--n-patients", type=int, required=True)
    ap.add_argument("--frequency", choices=["sample", "midpoint", "fixed"], default="sample")
    ap.add_argument("--substitution", type=float, default=0.0)
    ap.add_argument("--reporting", type=float, default=1.0, help="probability a symptom is reported")
    ap.add_argument("--hpo-terms", default="hpo_terms.json")
    ap.add_argument("--label", default="TEST2", help="gene label given to every patient")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("-o", "--output-json", default="synthetic_patients.json")
    args = ap.parse_args()
    try:
        generate(args.disease, args.n_patients, args.proportions, args.seed, args.substitution,
                 args.hpo_terms, args.frequency, args.label, args.output_json, reporting=args.reporting)
    except ValueError as e:
        sys.exit(str(e))
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
