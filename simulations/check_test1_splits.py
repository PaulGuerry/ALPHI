#!/usr/local/bin/python3.14
"""
Re-runs given TEST 1 cohorts (by seed) through pairD_phenome.py and shows how the clusters
relate to the true subgroups, which symptoms separate the clusters, and pairD's few-feature
flag (pairD_phenome.py of 261029 or later). For over-splits: is a true subgroup cut in two by
one isolated symptom?

The cohort is regenerated exactly as run_test1_grid.py did (same table, disease, N, k,
contrast, layout, substitution and seed; the seed is in test1_grid_iterations.csv).

Usage (in a TEST 1 folder, e.g. ~/pairD_eval/test1/min_dist_0/C2, with the generator,
the table and hpo_terms.json at hand):
  PGM_DISTANCE=dphenome PGM_ORGAN_MODE=nearest python3.14 check_test1_splits.py \
      -t test1_table_C2.csv --disease D2S2 -n 360 -k 3 --layout dispersed --substitution 0 \
      --seeds 733035982 1170169058 1121097979 1382909290 627750396 2133569852
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import generate_test1_patients as gt

CONTRASTS = {"strong": (0.80, 0.05), "weak": (0.60, 0.15)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-t", "--table", required=True)
    ap.add_argument("--disease", required=True, choices=["D1S0", "D2S2", "D3S3"])
    ap.add_argument("-n", type=int, required=True)
    ap.add_argument("-k", type=int, required=True)
    ap.add_argument("--contrast", default="strong", choices=list(CONTRASTS))
    ap.add_argument("--reporting", type=float, default=1.0)
    ap.add_argument("--layout", default="dispersed", choices=["dispersed", "concentrated"])
    ap.add_argument("--substitution", type=float, default=0.0)
    ap.add_argument("--hpo-terms", default="hpo_terms.json")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--paird-args", default="")
    args = ap.parse_args()

    exe = shutil.which("pairD_phenome.py")
    if not exe:
        sys.exit("pairD_phenome.py not found on PATH.")
    names = json.load(open(args.hpo_terms, encoding="utf-8"))["terms"]
    nm = lambda c: names.get("HP:" + str(c).replace("HP:", "").zfill(7), {}).get("name", c)
    high, low = CONTRASTS[args.contrast]
    table = str(Path(args.table).resolve())
    print(f"Distance {os.environ.get('PGM_DISTANCE', 'dphenome')}, organ mode {os.environ.get('PGM_ORGAN_MODE', 'nearest')}")

    for seed in args.seeds:
        with tempfile.TemporaryDirectory() as tmp:
            synth = Path(tmp) / "synthetic_patients.json"
            gt.generate(table, args.disease, args.n, args.k, high, low, null_like="D3S3", seed=seed,
                        output_json=str(synth), verbose=False, reporting=args.reporting, layout=args.layout,
                        substitution=args.substitution, hpo_terms=str(Path(args.hpo_terms).resolve()))
            out = Path(tmp) / "clusters.json"
            r = subprocess.run([exe, "-f", str(synth), "--gene", args.disease, "--skip-evaluation",
                                "--output-json", str(out)] + args.paird_args.split(),
                               cwd=tmp, capture_output=True, text=True)
            if r.returncode != 0 or not out.exists():
                print(f"\nseed {seed}: pairD_phenome.py failed\n{r.stdout[-600:]}\n{r.stderr[-600:]}")
                continue
            patients = {p["id"]: p for p in json.load(open(synth))["patients"]}
            res = json.load(open(out))["clustering_results"]["optimal"]["clusters"]
        assign, flags = {}, {}
        for c in res:
            name = "noise" if c["cluster_id"] == -1 else f"C{c['cluster_id']}"
            for p in c.get("patients") or []:
                assign[p.get("id") if isinstance(p, dict) else p] = name
            if c["cluster_id"] != -1:
                flags[name] = (c.get("flag_few_features"), c.get("n_discriminating_features"),
                               c.get("closest_cluster"),
                               [(nm(d["hpo_code"]), d.get("cluster"), d.get("closest_cluster")) for d in c.get("top_discriminating") or []])
        clusters = sorted(set(assign.values()))
        print(f"\n=== seed {seed}: {len([c for c in clusters if c != 'noise'])} clusters")
        print(f"    {'true':<10}" + "".join(f"{c:>8}" for c in clusters))
        for sg in sorted({p["subgroup"] for p in patients.values()}):
            cnt = Counter(assign.get(pid, "noise") for pid, p in patients.items() if p["subgroup"] == sg)
            print(f"    {'subgroup ' + str(sg):<10}" + "".join(f"{cnt.get(c, 0):>8}" for c in clusters))
        for c, (fl, n, partner, top) in flags.items():
            if fl is None:
                continue
            tops = "; ".join(f"{t} {a:.0%} vs {b:.0%}" for t, a, b in top)
            print(f"    {c} vs C{partner}: {n} discriminating symptom(s)"
                  f"{'  FLAGGED' if fl else ''} | {tops}")


if __name__ == "__main__":
    main()
