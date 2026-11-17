#!/usr/local/bin/python3.14
"""
TEST 1, post-hoc check: does the information content (IC) of the signature symptoms
explain why Resnik performs differently in compositions C1-C3?

For every signature symptom of each test1_table_C*.csv (per layout), and for the core
symptoms, this computes with the same annotation-based IC as pgmodules.py
(-log(n/N) / log(N), n = diseases annotated with the term or a descendant in
phenotype.hpoa, NOT annotations and non-phenotype aspects excluded):

  ic          IC of the symptom: the Resnik credit when two patients share it exactly.
  background  highest Resnik similarity (IC of the most informative common ancestor)
              between the symptom and any symptom of the other subgroups' signature sets or
              of the core. It is what a patient WITHOUT the symptom still earns for it in
              the best-match average.
  margin      ic - background: the Resnik signal the symptom contributes to separating
              its subgroup from the others.
  parent_ic   mean IC of the symptom's parents: the credit left when one report has been
              substituted by a parent term (2/3 of substitutions).

The summary gives the mean of each quantity over signature ranks 1..k (k = 3, 6, 9: the
symptoms a grid point with signature size k uses), per composition and layout, and the
mean IC of the core. With --results it adds Resnik's and OB's mean correct-k rate at the
same k, layout and p = 0 from the grid results, for comparison.

Usage (from ~/test1):
  python3.14 resnik_ic_check.py --hpoa ~/GITHUB/pairD_phenome/phenotype.hpoa \
      --tables C1/test1_table_C1.csv C2/test1_table_C2.csv C3/test1_table_C3.csv \
      --hpo-terms C1/hpo_terms.json --results C*/test1_grid_results.csv
  -o resnik_ic_check.csv writes the per-symptom values.
"""
import argparse
import csv
import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

SETS = ["A", "B", "C"]
K_VALUES = [3, 6, 9]
SUBSTITUTION_FOR_RESULTS = 0.0


# ---- HPO and IC ---------------------------------------------------------------------

def load_hpo(path):
    d = json.load(open(path, encoding="utf-8"))
    return d["terms"], d.get("alt_id_map", {})


def make_ancestors(terms):
    cache = {}

    def anc(t):
        hit = cache.get(t)
        if hit is None:
            hit = {t}
            for p in terms.get(t, {}).get("parents", []):
                hit |= anc(p)
            cache[t] = hit = frozenset(hit)
        return hit
    return anc


def annotation_ic(hpoa, terms, alt, anc, dbs=()):
    """{term: IC scaled to 0..1}, as in pgmodules._resnik_tables (annotation mode)."""
    diseases = defaultdict(set)
    with open(hpoa, encoding="utf-8") as f:
        rows = (line for line in f if not line.startswith("#"))
        reader = csv.reader(rows, delimiter="\t")
        col = {h: i for i, h in enumerate(next(reader))}
        need = max(col["database_id"], col["hpo_id"], col["aspect"], col["qualifier"])
        for r in reader:
            if len(r) <= need or r[col["aspect"]] != "P" or r[col["qualifier"]].strip().upper() == "NOT":
                continue
            if dbs and r[col["database_id"]].split(":")[0] not in dbs:
                continue
            code = r[col["hpo_id"]]
            diseases[r[col["database_id"]]].add(alt.get(code, code))
    n_dis = len(diseases)
    if n_dis == 0:
        sys.exit(f"No usable annotations in {hpoa}.")
    counts = defaultdict(int)
    for codes in diseases.values():
        closure = set()
        for c in codes:
            closure |= anc(c)
        for a in closure:
            counts[a] += 1
    ic_max = math.log(n_dis)
    ic = {t: -math.log(max(counts.get(t, 0), 1) / n_dis) / ic_max
          for t, v in terms.items() if not v.get("obsolete")}
    return ic, n_dis


# ---- tables -------------------------------------------------------------------------

def load_table(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    comp = Path(path).stem.split("_")[-1]
    core = [r for r in rows if r["role"] == "core"]
    layouts = sorted({r.get("layout") or "dispersed" for r in rows if r["role"] == "signature"})
    sig = {}
    for lay in layouts:
        sig[lay] = {st: sorted((r for r in rows if r["role"] == "signature" and r["set"] == st
                                and (r.get("layout") or "dispersed") == lay),
                               key=lambda r: int(r["rank"])) for st in SETS}
    return comp, core, sig


def read_results(patterns):
    """{(composition, disease, layout, k, distance): mean correct-k} at p = 0."""
    acc = defaultdict(list)
    for pat in patterns:
        for path in glob.glob(pat):
            for r in csv.DictReader(open(path, encoding="utf-8")):
                if r.get("status") != "ok" or r["disease"] == "D1S0":
                    continue
                if abs(float(r["substitution"]) - SUBSTITUTION_FOR_RESULTS) > 1e-9:
                    continue
                if r["distance"] not in ("dphenome", "resnik"):
                    continue
                acc[(r["composition"], r["disease"], r["layout"], int(r["sig_size"]), r["distance"])].append(
                    float(r["k_correct_mean"]))
    return {k: sum(v) / len(v) for k, v in acc.items()}


# ---- main ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", nargs="+", default=None,
                    help="test1_table_C*.csv (default: C*/test1_table_C*.csv here)")
    ap.add_argument("--hpoa", required=True, help="phenotype.hpoa used by pgmodules.py")
    ap.add_argument("--hpo-terms", default="hpo_terms.json")
    ap.add_argument("--db", default="", help="restrict diseases, e.g. ORPHA or OMIM,ORPHA (as PGM_HPOA_DB)")
    ap.add_argument("--results", nargs="*", default=[], help="test1_grid_results.csv files (globs allowed)")
    ap.add_argument("-o", "--output", default=None, help="per-symptom CSV")
    args = ap.parse_args()

    tables = args.tables or sorted(glob.glob("C*/test1_table_C*.csv"))
    if not tables:
        sys.exit("No tables given or found (C*/test1_table_C*.csv).")
    terms, alt = load_hpo(args.hpo_terms)
    anc = make_ancestors(terms)
    dbs = {x.strip() for x in args.db.split(",") if x.strip()}
    ic, n_dis = annotation_ic(args.hpoa, terms, alt, anc, dbs)
    print(f"Annotation IC from {args.hpoa}: {n_dis} diseases{' (' + args.db + ')' if dbs else ''}.\n")

    def sim(a, b):
        return max((ic.get(c, 0.0) for c in anc(a) & anc(b)), default=0.0)

    per_symptom, summary = [], []
    for path in tables:
        comp, core, sig = load_table(path)
        core_codes = [r["hpo"] for r in core]
        core_ic = sum(ic.get(c, 0.0) for c in core_codes) / len(core_codes)
        for lay, sets in sig.items():
            for st in SETS:
                others = core_codes + [r["hpo"] for o in SETS if o != st for r in sets[o]]
                for r in sets[st]:
                    t = r["hpo"]
                    bg_code = max(others, key=lambda o: sim(t, o))
                    bg = sim(t, bg_code)
                    parents = [p for p in terms.get(t, {}).get("parents", []) if p in ic]
                    per_symptom.append(dict(
                        composition=comp, layout=lay, set=st, rank=int(r["rank"]), hpo=t, label=r["label"],
                        organ_system=r["organ_system"], ic=round(ic.get(t, 0.0), 4),
                        background=round(bg, 4), background_term=bg_code,
                        margin=round(ic.get(t, 0.0) - bg, 4),
                        parent_ic=round(sum(ic[p] for p in parents) / len(parents), 4) if parents else ""))
            for k in K_VALUES:
                rows = [x for x in per_symptom if x["composition"] == comp and x["layout"] == lay
                        and x["rank"] <= k]
                if not rows:
                    continue
                pics = [x["parent_ic"] for x in rows if x["parent_ic"] != ""]
                summary.append(dict(
                    composition=comp, layout=lay, k=k, n=len(rows),
                    ic=sum(x["ic"] for x in rows) / len(rows),
                    background=sum(x["background"] for x in rows) / len(rows),
                    margin=sum(x["margin"] for x in rows) / len(rows),
                    parent_ic=sum(pics) / len(pics) if pics else float("nan"),
                    core_ic=core_ic))

    res = read_results(args.results) if args.results else {}
    head = f"{'comp':<5}{'layout':<14}{'k':>3}{'IC':>7}{'backgr':>8}{'margin':>8}{'parentIC':>9}{'coreIC':>8}"
    if res:
        head += "   correct-k at p = 0 (D2S2/D3S3):  Resnik      OB"
    print(head)
    for s in sorted(summary, key=lambda s: (s["k"], s["layout"], s["composition"])):
        line = (f"{s['composition']:<5}{s['layout']:<14}{s['k']:>3}{s['ic']:>7.3f}{s['background']:>8.3f}"
                f"{s['margin']:>8.3f}{s['parent_ic']:>9.3f}{s['core_ic']:>8.3f}")
        if res:
            def kc(dist):
                v = [res.get((s["composition"], d, s["layout"], s["k"], dist)) for d in ("D2S2", "D3S3")]
                return "/".join("  -" if x is None else f"{100 * x:3.0f}" for x in v)
            line += f"{'':>36}{kc('resnik'):>9}{kc('dphenome'):>9}"
        print(line)

    if args.output:
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(per_symptom[0]))
            w.writeheader()
            w.writerows(per_symptom)
        print(f"\nPer-symptom values: {args.output}")


if __name__ == "__main__":
    main()
