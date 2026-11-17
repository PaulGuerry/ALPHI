#!/usr/local/bin/python3.14
"""
TEST 3 (Simulation 3): can pairD_phenome.py (ALPHI) separate a large cohort of many
distinct diseases into one cluster per disease? Cohorts mix --n-per-disease patients of
each of the 44 syndromes of Kohler et al. (2009), simulated as in that article with
generate_test3_patients.py (independent features at the SI frequencies, sex-specific
features by sex, optional noise terms).

Grid (every combination is run --iterations times; no search)
  n_per_disease  20 by default (--n-per-disease 20 30 ...)
  noise          q = 0 and 0.5 (Kohler et al.: half as many unrelated terms as features)
  weighting      uniform (Kohler et al.) by default; annotated = noise terms weighted by
                 their number of disease annotations in phenotype.hpoa (--weighting
                 uniform annotated). Only meaningful when q > 0: at q = 0 one run serves both.
  distance       dphenome (OB), jaccard, resnik

Pairing: the generation seed depends only on n_per_disease and the iteration, so the
noise levels, weightings and distances are compared on the same underlying patients
(the noise has its own random stream in generate_test3_patients.py).

pairD_phenome.py runs in a fresh process for every iteration (distance passed through the
PGM_DISTANCE environment variable), so memory is released after each run; --in-process
runs it inside this process instead, as run_test2_grid.py does.

pairD_phenome.py is run with --min-symptoms 1 by default (--min-symptoms): the
simulated patients of sparse diseases (e.g. Holt-Oram, about 2 features) are kept.

Scores per iteration (the optimal clustering only)
  n_clusters   non-noise clusters found; k_correct 1 if it equals the number of diseases;
  k_error      n_clusters - number of diseases
  ari          adjusted Rand index, true diseases vs clusters, assigned patients only
  ari_all      the same over all generated patients, noise counted as one more group
  coverage     fraction of generated patients assigned to a cluster
  recovered    number of diseases recovered: a cluster holds at least --recall (50 %) of
               the disease's patients, and the disease makes up more than --purity (50 %)
               of that cluster (a strict majority, so a cluster recovers at most one disease)
  merged       clusters in which no disease is a strict majority
  split        diseases whose assigned patients are spread over 2 or more clusters with at
               least 25 % of the disease's patients each
Per disease and iteration (test3_diseases.csv): patients, mean features (without noise),
patients assigned, share of the disease's patients in its main cluster (recall), share of
that cluster made up by the disease (purity), recovered (0/1).

Output
  test3_iterations.csv  one row per iteration
  test3_points.csv      one row per grid point (means over iterations), rewritten as it goes
  test3_diseases.csv    one row per disease per iteration
--resume skips the iterations already in test3_iterations.csv.

Needs next to it: generate_test3_patients.py, generate_test2_patients.py,
run_test2_grid.py (pairD_phenome.py helpers), hpo_terms.json, the symptom lists from
parse_phenomizer_si.py and, for --weighting annotated or the resnik distance,
phenotype.hpoa; pairD_phenome.py on PATH with its pgmodules.py and hp_trees.json.

Usage:
  python3.14 run_test3_grid.py --diseases phenomizer/symptom_list_*.csv --dry-run
  python3.14 run_test3_grid.py --diseases phenomizer/symptom_list_*.csv 2>&1 | tee test3.log
  python3.14 run_test3_grid.py --diseases phenomizer/symptom_list_*.csv --noise 0.5 \
      --weighting uniform annotated --distance dphenome            # part of the grid
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import shlex
import shutil
import subprocess
import sys
import time
import traceback
import zlib
from pathlib import Path

import numpy as np

import generate_test3_patients as g3
import run_test2_grid as r2          # in-process pairD_phenome.py, distance switch, ARI

sys.stdout.reconfigure(line_buffering=True)

CLUSTER_JSON = r2.CLUSTER_JSON
SYNTH_JSON = "synthetic_patients.json"
LABEL = "TEST3"

N_PER_DISEASE = [20]
NOISE = [0.0, 0.5]
WEIGHTING = ["uniform"]
DISTANCES = ["dphenome", "jaccard", "resnik"]
N_ITER = 10

METRICS = ["n_clusters", "k_correct", "k_error", "ari", "ari_all", "coverage", "recovered", "merged", "split"]
AXES = ["n_diseases", "n_per_disease", "noise", "weighting", "distance"]
RAW_FIELDS = AXES + ["iteration", "seed", "n_patients"] + METRICS
POINT_FIELDS = AXES + ["iterations"] + [f"{m}_mean" for m in METRICS] + ["recovered_min", "n_clusters_min",
                                                                         "n_clusters_max"]
DISEASE_FIELDS = AXES + ["iteration", "disease", "n", "mean_features", "n_assigned", "recall", "purity",
                         "recovered"]


def point_seed(base, n_per, iteration):
    """Independent of the noise, weighting and distance (paired comparisons)."""
    return (zlib.crc32(f"test3|{n_per}|{iteration}".encode()) ^ base) & 0x7FFFFFFF


def score(n_true, recall_min, purity_min):
    """Iteration metrics and per-disease rows for the clustering in CLUSTER_JSON."""
    with open(CLUSTER_JSON, encoding="utf-8") as f:
        clusters = json.load(f)["clustering_results"]["optimal"]["clusters"]
    with open(SYNTH_JSON, encoding="utf-8") as f:
        patients = json.load(f)["patients"]
    truth = {str(p["id"]): p["subgroup"] for p in patients}
    real = [c for c in clusters if c.get("cluster_id") != -1]
    assigned = {}
    for c in real:
        for pid in (r2._cluster_members(c) or []):
            if pid in truth:
                assigned[pid] = c["cluster_id"]
    t = [truth[p] for p in assigned]
    pr = list(assigned.values())
    all_t = list(truth.values())
    all_p = [assigned.get(pid, "noise") for pid in truth]

    size = {}                                    # cluster -> patients
    cross = {}                                   # (disease, cluster) -> patients
    for pid, cl in assigned.items():
        size[cl] = size.get(cl, 0) + 1
        cross[(truth[pid], cl)] = cross.get((truth[pid], cl), 0) + 1
    by_disease = {}
    for p in patients:
        by_disease.setdefault(p["subgroup"], []).append(p)
    rows, n_rec, n_split = [], 0, 0
    for g, members in sorted(by_disease.items()):
        n = len(members)
        cells = sorted(((v, cl) for (d, cl), v in cross.items() if d == g), reverse=True)
        n_assigned = sum(v for v, _ in cells)
        main_v, main_cl = cells[0] if cells else (0, None)
        recall = main_v / n
        purity = main_v / size[main_cl] if main_cl is not None else float("nan")
        rec = bool(cells) and recall >= recall_min and purity > purity_min
        n_rec += rec
        n_split += sum(v >= 0.25 * n for v, _ in cells) >= 2
        rows.append(dict(disease=members[0]["disease"], n=n,
                         mean_features=float(np.mean([p.get("n_true_symptoms", len(p["symptoms"])) for p in members])),
                         n_assigned=n_assigned, recall=recall, purity=purity, recovered=int(rec)))
    merged = sum(max((v for (d, c2), v in cross.items() if c2 == cl), default=0) / size[cl] <= purity_min
                 for cl in size)
    m = {"n_clusters": float(len(real)), "k_correct": float(len(real) == n_true),
         "k_error": float(len(real) - n_true),
         "ari": r2.adjusted_rand(t, pr) if t else float("nan"),
         "ari_all": r2.adjusted_rand(all_t, all_p),
         "coverage": len(assigned) / len(truth) if truth else float("nan"),
         "recovered": float(n_rec), "merged": float(merged), "split": float(n_split)}
    return m, rows


def cluster_subprocess(paird_args, distance, args):
    """pairD_phenome.py in a fresh process (memory released after every run). The distance
    and organ mode reach pgmodules.py through PGM_DISTANCE and PGM_ORGAN_MODE."""
    exe = shutil.which("pairD_phenome.py")
    if not exe:
        print("[!] pairD_phenome.py not found on PATH.")
        return False
    env = dict(os.environ, PGM_DISTANCE=distance, PGM_ORGAN_MODE=args.organ_mode)
    if args.hpoa and Path(args.hpoa).exists():
        env.setdefault("PGM_HPOA", str(Path(args.hpoa).resolve()))
    r = subprocess.run([sys.executable, os.path.realpath(exe)] + list(paird_args), env=env)
    if r.returncode != 0:
        print(f"[!] pairD_phenome.py exited with code {r.returncode}")
    return r.returncode == 0


def run_iteration(files, n_per, noise, weighting, distance, seed, args):
    before = r2._paird_result_dirs()
    try:
        if CLUSTER_JSON.exists():
            CLUSTER_JSON.unlink()
        try:
            g3.generate(files, n_per, noise, seed=seed, hpo_terms=args.hpo_terms, label=LABEL,
                        output_json=SYNTH_JSON, verbose=True, noise_weighting=weighting, hpoa=args.hpoa)
        except Exception as e:
            print(f"[!] Patient generation failed: {e}")
            traceback.print_exc()
            return r2._record(None)
        paird_args = (["-f", SYNTH_JSON, "--gene", LABEL, "--skip-evaluation", "--output-json", str(CLUSTER_JSON),
                       "--min-symptoms", str(args.min_symptoms)] + shlex.split(args.paird_args))
        if args.in_process:
            r2.set_distance(distance)
            ok = r2._cluster_in_process(paird_args)
        else:
            ok = cluster_subprocess(paird_args, distance, args)
        if not ok or not CLUSTER_JSON.exists():
            print("[!] pairD_phenome clustering failed. Skipping iteration.")
            return r2._record(None)
        try:
            res = score(len(files), args.recall, args.purity)
        except Exception as e:
            print(f"[!] Could not score the clustering: {e}")
            traceback.print_exc()
            return r2._record(None)
        return r2._record(res)
    finally:
        r2._remove_new_result_dirs(before)


def key(row):
    return (int(row["n_per_disease"]), float(row["noise"]), row["weighting"], row["distance"], int(row["iteration"]))


def write_points(path, raw_rows):
    groups = {}
    for r in raw_rows:
        groups.setdefault(tuple(r[a] for a in AXES), []).append(r)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=POINT_FIELDS)
        w.writeheader()
        for k, rs in groups.items():
            row = dict(zip(AXES, k), iterations=len(rs))
            for m in METRICS:
                row[f"{m}_mean"] = float(np.nanmean([float(r[m]) for r in rs]))
            row["recovered_min"] = min(float(r["recovered"]) for r in rs)
            row["n_clusters_min"] = min(float(r["n_clusters"]) for r in rs)
            row["n_clusters_max"] = max(float(r["n_clusters"]) for r in rs)
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diseases", nargs="+", required=True, help="symptom lists from parse_phenomizer_si.py")
    ap.add_argument("--n-per-disease", nargs="+", type=int, default=N_PER_DISEASE)
    ap.add_argument("--noise", nargs="+", type=float, default=NOISE)
    ap.add_argument("--weighting", nargs="+", choices=["uniform", "annotated"], default=WEIGHTING)
    ap.add_argument("--distance", nargs="+", default=DISTANCES, choices=r2.DISTANCE_CHOICES)
    ap.add_argument("--iterations", type=int, default=N_ITER)
    ap.add_argument("--min-symptoms", type=int, default=1, help="passed to pairD_phenome.py (its default is 5)")
    ap.add_argument("--recall", type=float, default=0.5, help="recovered: share of the disease in its cluster")
    ap.add_argument("--purity", type=float, default=0.5, help="recovered: share of the cluster from the disease (strictly more than)")
    ap.add_argument("--hpo-terms", default="hpo_terms.json")
    ap.add_argument("--hpoa", default="phenotype.hpoa", help="for --weighting annotated")
    ap.add_argument("--base-seed", type=int, default=1)
    ap.add_argument("--organ-mode", choices=["nearest", "penalty"], default="nearest")
    ap.add_argument("--raw", default="test3_iterations.csv")
    ap.add_argument("-o", "--output", default="test3_points.csv")
    ap.add_argument("--per-disease", default="test3_diseases.csv")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--keep-results", action="store_true")
    ap.add_argument("--in-process", action="store_true",
                    help="run pairD_phenome.py inside this process, as run_test2_grid.py does (faster start, "
                         "but memory accumulates over the grid); default: a fresh process per run")
    ap.add_argument("--paird-args", default="",
                    help='Extra options passed to pairD_phenome.py, e.g. "--gridsearch 3 --min-dist 0". '
                         'Use a separate folder (or output names) per setting: they are not part of the --resume key.')
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = sorted(args.diseases)
    for f in files:
        if not Path(f).exists():
            sys.exit(f"{f} not found.")
    if not Path(args.hpo_terms).exists():
        sys.exit(f"{args.hpo_terms} not found.")
    if "annotated" in args.weighting and any(q > 0 for q in args.noise) and not Path(args.hpoa).exists():
        sys.exit(f"{args.hpoa} not found (needed for --weighting annotated).")
    combos = []
    for n_per in args.n_per_disease:
        for q in args.noise:
            for wt in (args.weighting if q > 0 else ["uniform"]):    # no noise: weighting irrelevant
                for d in args.distance:
                    combos.append((n_per, q, wt, d))
    print(f"=== TEST 3: {len(files)} diseases; {len(combos)} grid points x {args.iterations} iterations "
          f"= {len(combos) * args.iterations} runs ===")
    for n_per, q, wt, d in combos:
        print(f"  {n_per} per disease ({n_per * len(files)} patients), noise {q:g}"
              + (f" {wt}" if q > 0 else "") + f", {d}")
    print(f"pairD_phenome.py --min-symptoms {args.min_symptoms}" + (f" {args.paird_args}" if args.paird_args else ""))
    if args.dry_run:
        return

    os.environ["PGM_ORGAN_MODE"] = args.organ_mode
    try:
        mp.set_start_method("fork", force=True)
    except (RuntimeError, ValueError) as e:
        print(f"[!] Could not set multiprocessing start method to 'fork': {e}")
    r2.KEEP_RESULTS = args.keep_results
    if args.in_process:
        for d in args.distance:
            r2.set_distance(d)                           # checks pgmodules.py (and phenotype.hpoa) now
    else:
        exe = shutil.which("pairD_phenome.py")
        if not exe:
            sys.exit("pairD_phenome.py not found on PATH.")
        pgm = Path(os.path.realpath(exe)).with_name("pgmodules.py")
        if "PGM_DISTANCE" not in pgm.read_text(encoding="utf-8"):
            sys.exit(f"{pgm} does not read PGM_DISTANCE: install the updated pgmodules.py.")
        print(f"[*] pairD_phenome.py run in a fresh process per iteration: {os.path.realpath(exe)}")

    raw, dis, pts = Path(args.raw), Path(args.per_disease), Path(args.output)
    raw_rows = list(csv.DictReader(open(raw, newline="", encoding="utf-8"))) if args.resume and raw.exists() else []
    done = {key(r) for r in raw_rows}
    t0 = time.perf_counter()
    with open(raw, "a" if raw_rows else "w", newline="", encoding="utf-8") as f_raw, \
            open(dis, "a" if raw_rows and dis.exists() else "w", newline="", encoding="utf-8") as f_dis:
        w_raw = csv.DictWriter(f_raw, fieldnames=RAW_FIELDS, extrasaction="ignore")
        w_dis = csv.DictWriter(f_dis, fieldnames=DISEASE_FIELDS, extrasaction="ignore")
        if f_raw.tell() == 0:
            w_raw.writeheader()
        if f_dis.tell() == 0:
            w_dis.writeheader()
        for i, (n_per, q, wt, d) in enumerate(combos, start=1):
            axes = dict(n_diseases=len(files), n_per_disease=n_per, noise=q, weighting=wt if q > 0 else "", distance=d)
            label = f"{n_per}/disease, noise {q:g}" + (f" {wt}" if q > 0 else "") + f", {d}"
            print("\n" + "=" * 80 + f"\n[point {i}/{len(combos)}] {label}\n" + "=" * 80)
            for it in range(1, args.iterations + 1):
                if (n_per, q, axes["weighting"], d, it) in done:
                    print(f"[=] iteration {it}: in {raw}")
                    continue
                seed = point_seed(args.base_seed, n_per, it)
                print(f"--- Iteration {it} (seed {seed}) ---")
                t1 = time.perf_counter()
                res = run_iteration(files, n_per, q, wt, d, seed, args)
                if not res:
                    continue
                m, rows = res
                row = {**axes, "iteration": it, "seed": seed, "n_patients": n_per * len(files), **m}
                w_raw.writerow(row)
                f_raw.flush()
                raw_rows.append({k: str(v) for k, v in row.items()})
                for r in rows:
                    w_dis.writerow({**axes, "iteration": it, **r})
                f_dis.flush()
                print(f"    clusters {int(m['n_clusters'])} of {len(files)} | recovered {int(m['recovered'])} "
                      f"| merged {int(m['merged'])} | split {int(m['split'])} | ARI {m['ari']:.3f} "
                      f"| ARI all {m['ari_all']:.3f} | coverage {m['coverage']:.0%} ({time.perf_counter() - t1:.0f} s)")
                write_points(pts, raw_rows)
    print(f"\n[Complete] TEST 3 finished in {time.perf_counter() - t0:.0f} s. "
          f"Results: {raw} (per iteration), {pts} (per grid point), {dis} (per disease).")


if __name__ == "__main__":
    main()
