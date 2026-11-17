#!/usr/local/bin/python3.14
"""
Do the spurious splits of a single-disease TEST 2 control separate patients by SEX?

For each given seed, regenerates exactly the cohort the grid used (same disease file,
N, seed, reporting and substitution; generate_test2_patients.py), clusters it with
pairD_phenome.py (distance from PGM_DISTANCE / PGM_ORGAN_MODE, as in the grid) and
cross-tabulates the clusters against the patients' sex. It also lists the symptoms that
differ most between the clusters.

Usage (in the TEST 2 folder, e.g. ~/calib/md0/t2):
  PGM_DISTANCE=dphenome PGM_ORGAN_MODE=nearest python3.14 check_split_by_sex.py \
      --disease symptom_list_wilson.txt -n 180 \
      --seeds 1932052347 1950679906 1157132229 720768745 882023242 1025824609
The seeds of the runs that split are in test2_iterations.csv (column seed).
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

import generate_test2_patients as gt

LABEL = "TEST2"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disease", required=True)
    ap.add_argument("-n", type=int, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--reporting", type=float, default=1.0)
    ap.add_argument("--substitution", type=float, default=0.0)
    ap.add_argument("--hpo-terms", default="hpo_terms.json")
    ap.add_argument("--paird-args", default="", help="extra pairD_phenome.py options, as in the grid")
    ap.add_argument("--top", type=int, default=6)
    args = ap.parse_args()

    exe = shutil.which("pairD_phenome.py")
    if not exe:
        sys.exit("pairD_phenome.py not found on PATH.")
    H = gt.hpo(args.hpo_terms)
    disease = str(Path(args.disease).resolve())
    print(f"Distance {os.environ.get('PGM_DISTANCE', 'dphenome')}, organ mode {os.environ.get('PGM_ORGAN_MODE', 'nearest')}")

    for seed in args.seeds:
        with tempfile.TemporaryDirectory() as tmp:
            synth = Path(tmp) / "synthetic_patients.json"
            gt.generate([disease], args.n, [1], seed=seed, substitution=args.substitution,
                        hpo_terms=args.hpo_terms, label=LABEL, output_json=str(synth), verbose=False,
                        reporting=args.reporting)
            out = Path(tmp) / "clusters.json"
            cmd = [exe, "-f", str(synth), "--gene", LABEL, "--skip-evaluation", "--output-json", str(out)] \
                + args.paird_args.split()
            r = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True)
            if r.returncode != 0 or not out.exists():
                print(f"\nseed {seed}: pairD_phenome.py failed\n{r.stdout[-800:]}\n{r.stderr[-800:]}")
                continue
            patients = {p["id"]: p for p in json.load(open(synth))["patients"]}
            res = json.load(open(out))["clustering_results"]["optimal"]["clusters"]
            assign = {}
            for c in res:
                for p in c.get("patients") or []:
                    pid = p.get("id") if isinstance(p, dict) else p
                    assign[pid] = "noise" if c["cluster_id"] == -1 else f"C{c['cluster_id']}"
        clusters = sorted(set(assign.values()))
        print(f"\n=== seed {seed}: {len(clusters)} groups ({', '.join(clusters)})")
        print(f"    {'':<8}" + "".join(f"{c:>8}" for c in clusters))
        for sex in ("F", "M"):
            cnt = Counter(assign[pid] for pid, p in patients.items() if p["sex"] == sex and pid in assign)
            print(f"    {sex:<8}" + "".join(f"{cnt.get(c, 0):>8}" for c in clusters))
        real = [c for c in clusters if c != "noise"]
        if len(real) >= 2:
            prev = {}
            for c in real:
                mem = [patients[pid] for pid in assign if assign[pid] == c]
                cnt = Counter(s for p in mem for s in p["symptoms"])
                prev[c] = {s: v / len(mem) for s, v in cnt.items()}
            terms = set().union(*prev.values())
            diff = sorted(terms, key=lambda s: -(max(prev[c].get(s, 0) for c in real) - min(prev[c].get(s, 0) for c in real)))
            print("    symptoms differing most between clusters: " + "; ".join(
                f"{H.terms.get('HP:' + s, {}).get('name', s)} " + "/".join(f"{prev[c].get(s, 0):.0%}" for c in real)
                for s in diff[: args.top]))


if __name__ == "__main__":
    main()
