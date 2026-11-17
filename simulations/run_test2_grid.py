#!/usr/local/bin/python3.14
"""
TEST 2: how many patients does pairD_phenome.py need to separate real diseases that are
known to be distinct? Cohorts mix patients of Marfan, Wilson and Alagille (by default),
each simulated from its Orphanet symptom list with generate_test2_patients.py.

Scenarios (--kinds; group sizes are always unequal)
  control  one disease alone (correct answer: 1 cluster), at N = 10, 30, 90, 180 (or
           --control-n); reports how often a real, heterogeneous symptom profile is split
           spuriously. Not searched: each N is run once with the usual iterations.
  pair     two diseases in proportions --pair-ratio (default 2:1), both ways round
           (A 2 : B 1 and B 2 : A 1), since which disease is the minority matters
  triple   the three diseases in proportions --triple-ratio (default 3:2:1), in the three
           rotations, so each disease is once the largest, once the middle and once the
           smallest group

Search for the N needed (pairs and triples)
  For each scenario, reporting rate, substitution level and distance, the smallest
  cohort size on the ladder --ladder at which the true number of clusters is found in
  at least --target (default 80 %) of the runs. Bisection on the ladder (default
  10 20 30 45 60 90 120 180), starting at --start (default 60), then halving the
  remaining interval below (if N passes) or above (if it fails). Assumes that the rate
  rises with N; --full evaluates every rung instead, giving the whole curve. A search
  that fails at the top of the ladder is reported as not reached (N needed > 180).
  Each evaluated N runs --iter-initial iterations, extended to --iter-max when the rate is
  within 2 standard errors of the target.

Other axes
  substitution  p = 0, 0.25, 0.5: imprecise reporting (parent or sibling term)
  distance      dphenome (OB), jaccard, resnik
  reporting     r = 1 by default: probability that a symptom a patient has is reported
  Orphanet frequency classes are sampled within their range for every cohort
  (--frequency sample, see generate_test2_patients.py).

Pairing: the generation seed depends only on the scenario, N and the iteration, not on p,
r or the distance, so those are compared on the same underlying patients wherever the
searches evaluate the same N.

Scores per iteration (the optimal clustering only)
  n_clusters  non-noise clusters found;  k_correct 1 if it equals the number of diseases
  ari         adjusted Rand index between true diseases and clusters, assigned patients only
  ari_all     the same over all generated patients, noise counted as one more group
              (ARI is left empty for the controls: undefined with one true group)
  coverage    fraction of generated patients assigned to a cluster (pairD_phenome.py also
              leaves out patients with fewer than its --min-symptoms, default 5)

Output
  test2_points.csv      one row per evaluated (scenario, r, p, distance, N)
  test2_iterations.csv  one row per iteration
  test2_n_needed.csv    one row per search: the N needed (empty if not reached on the ladder)
--resume reuses the points already in test2_points.csv (searches continue from them).

Needs next to it: generate_test2_patients.py, hpo_terms.json and the Orphanet symptom lists;
pairD_phenome.py on PATH with its pgmodules.py (distance switch), hp_trees.json and, for
resnik, phenotype.hpoa.

Usage:
  python3.14 run_test2_grid.py --dry-run
  python3.14 run_test2_grid.py 2>&1 | tee test2.log
  python3.14 run_test2_grid.py --kinds pair --substitution 0 --distance dphenome   # part of the grid
"""
import argparse
import shlex
import csv
import importlib.util
import itertools
import json
import math
import multiprocessing as mp
import os
import shutil
import sys
import time
import traceback
import zlib
from pathlib import Path

import numpy as np

import generate_test2_patients as gt

sys.stdout.reconfigure(line_buffering=True)

# pairD_phenome.py writes its results here (--output-json); deleted before each run.
CLUSTER_JSON = Path("cluster_visualization_current.json")
CLUSTER_MEMBER_KEYS = ("patients", "patient_ids", "members", "patient_list")
SYNTH_JSON = "synthetic_patients.json"
LABEL = "TEST2"

MAX_CONSECUTIVE_FAILURES = 10   # stop the grid if this many iterations fail in a row
_consecutive_failures = 0
_PAIRD_MODULE = None
KEEP_RESULTS = False
_RESNIK_REPORTED = []

# =============================================================================
# pairD_phenome.py in-process, distance switch, ARI (copied unchanged from run_test1_grid.py)
# =============================================================================

def _cluster_members(cluster: dict):
    for key in CLUSTER_MEMBER_KEYS:
        val = cluster.get(key)
        if isinstance(val, list) and val:
            ids = []
            for p in val:
                pid = (p.get("id") or p.get("patient_id")) if isinstance(p, dict) else p
                if pid is not None:
                    ids.append(str(pid))
            return ids
    return None


def _record(result):
    """Counts consecutive failed iterations and aborts the grid if the setup is clearly broken."""
    global _consecutive_failures
    _consecutive_failures = 0 if result else _consecutive_failures + 1
    if _consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
        print(f"\n[!] HALT: {MAX_CONSECUTIVE_FAILURES} iterations failed in a row. "
              "Fix the error shown above, then restart with --resume.")
        sys.exit(1)
    return result


def _load_paird():
    """Imports pairD_phenome.py once, from the location 'which pairD_phenome.py' points to."""
    global _PAIRD_MODULE
    if _PAIRD_MODULE is None:
        exe = shutil.which("pairD_phenome.py")
        if not exe:
            raise FileNotFoundError("pairD_phenome.py not found on PATH.")
        real = os.path.realpath(exe)                    # follow the ~/.local/bin symlink
        script_dir = os.path.dirname(real)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)              # so 'import pgmodules' works
        spec = importlib.util.spec_from_file_location("pairD_phenome", real)
        module = importlib.util.module_from_spec(spec)
        sys.modules["pairD_phenome"] = module           # needed by its worker processes
        spec.loader.exec_module(module)
        print(f"[*] Loaded pairD_phenome.py in-process from {real}")
        # Make sure the distance setting requested for the grid is the one in effect
        pgm = sys.modules.get("pgmodules")
        wanted = os.environ.get("PGM_ORGAN_MODE", "nearest")
        actual = getattr(pgm, "ORGAN_MODE", None)
        if actual != wanted:
            raise RuntimeError(
                f"pgmodules.py in use ({getattr(pgm, '__file__', '?')}) has organ mode {actual!r}, "
                f"but the grid requested {wanted!r}. Install the updated pgmodules.py next to pairD_phenome.py.")
        print(f"[*] dPhenome organ-system handling: {actual} ({pgm.__file__})")
        _PAIRD_MODULE = module
    return _PAIRD_MODULE


def _cluster_in_process(paird_args) -> bool:
    """
    Calls pairD_phenome.main() with the given arguments. pairD_phenome.py redirects
    sys.stdout/sys.stderr to a log file and never restores or closes it, so both are
    restored and the log closed afterwards (otherwise open files pile up over the grid).
    """
    module = _load_paird()
    saved_argv, saved_out, saved_err = sys.argv, sys.stdout, sys.stderr
    sys.argv = ["pairD_phenome.py"] + list(paird_args)
    ok = True
    try:
        module.main()
    except SystemExit as e:
        ok = e.code in (None, 0)
        if not ok:
            print(f"[!] pairD_phenome.py exited with code {e.code}", file=saved_out)
    except Exception as e:
        ok = False
        print(f"[!] EXCEPTION in pairD_phenome.py: {e}", file=saved_out)
        traceback.print_exc(file=saved_out)
    finally:
        for stream in (sys.stdout, sys.stderr):
            log = getattr(stream, "log", None)
            if log is not None and not log.closed:
                try:
                    log.close()
                except Exception:
                    pass
        sys.argv, sys.stdout, sys.stderr = saved_argv, saved_out, saved_err
    return ok


def _paird_result_dirs() -> set:
    """pairD_phenome.py's per-run folders: <yymmdd>_<genes>_<system>_<pairs> containing its outputs."""
    found = set()
    for p in Path('.').iterdir():
        if p.is_dir() and p.name[:6].isdigit() and p.name[6:7] == "_" and \
                ((p / "cdm.csv").exists() or (p / "pipeline.log").exists()):
            found.add(p.resolve())
    return found


def _remove_new_result_dirs(before: set):
    if KEEP_RESULTS:
        return
    for d in _paird_result_dirs() - before:
        shutil.rmtree(d, ignore_errors=True)


def set_distance(mode):
    """Distance used by pgmodules.dPhenome() in the next pairD_phenome.py run."""
    _load_paird()
    pgm = sys.modules["pgmodules"]
    if not hasattr(pgm, "DISTANCE_MODE") or mode not in getattr(pgm, "DISTANCE_MODES", ()):
        raise RuntimeError(f"pgmodules.py in use ({pgm.__file__}) does not support distance {mode!r}: "
                           "install the updated pgmodules.py next to pairD_phenome.py.")
    if mode == "resnik":
        desc = pgm.resnik_ic_description()      # raises if phenotype.hpoa is missing
        if not _RESNIK_REPORTED:
            print(f"[*] Resnik distance with {desc}")
            _RESNIK_REPORTED.append(True)
    pgm.DISTANCE_MODE = mode


def adjusted_rand(true, pred):
    """Adjusted Rand index (Hubert & Arabie). 1 = identical partitions, ~0 = chance."""
    from math import comb
    true, pred = list(true), list(pred)
    n = len(true)
    if n < 2:
        return float("nan")
    cont = {}
    for t, p in zip(true, pred):
        cont[(t, p)] = cont.get((t, p), 0) + 1
    a, b = {}, {}
    for (t, p), v in cont.items():
        a[t] = a.get(t, 0) + v
        b[p] = b.get(p, 0) + v
    sum_ij = sum(comb(v, 2) for v in cont.values())
    sum_a = sum(comb(v, 2) for v in a.values())
    sum_b = sum(comb(v, 2) for v in b.values())
    expected = sum_a * sum_b / comb(n, 2)
    max_index = (sum_a + sum_b) / 2
    if max_index == expected:          # both partitions trivial (e.g. one class, one cluster)
        return 1.0
    return (sum_ij - expected) / (max_index - expected)


# =============================================================================
# TEST 2
# =============================================================================

DISEASE_FILES = ["symptom_list_marfan.txt", "symptom_list_wilson.txt", "symptom_list_alagille.txt"]
KINDS = ["control", "pair", "triple"]
LADDER = [10, 20, 30, 45, 60, 90, 120, 180]
CONTROL_N = [10, 30, 90, 180]
START_N = 60
SUBSTITUTION = [0.0, 0.25, 0.5]
REPORTING = [1.0]
DISTANCES = ["dphenome", "jaccard", "resnik"]
DISTANCE_CHOICES = ["dphenome", "jaccard", "resnik", "hamming"]
N_ITER_INITIAL = 10
N_ITER_MAX = 20
CLOSE_CALL_SE = 2.0
MIN_SUCCESSFUL = 3

METRICS = ["n_clusters", "k_correct", "ari", "ari_all", "coverage"]
POINT_AXES = ["kind", "scenario", "diseases", "proportions", "reporting", "substitution", "distance", "n_patients"]
POINT_FIELDS = (POINT_AXES + ["n_true", "status", "iterations_run", "successful_iterations", "close_call",
                              "rate", "passed"] + [f"{m}_mean" for m in METRICS] + ["n_clusters_max"])
RAW_FIELDS = POINT_AXES + ["iteration", "seed"] + METRICS
SEARCH_FIELDS = ["kind", "scenario", "diseases", "proportions", "reporting", "substitution", "distance",
                 "target", "n_needed", "reached", "ladder_max", "evaluated"]


def fmt_ratio(w):
    return ":".join(f"{x:g}" for x in w)


def build_scenarios(files, kinds, pair_ratio, triple_ratio):
    """[{kind, id, files, names, proportions}]"""
    names = {f: gt.disease_name(f) for f in files}
    out = []

    def add(kind, fs, w):
        sid = "+".join(f"{names[f]}{x:g}" for f, x in zip(fs, w)) if len(fs) > 1 else names[fs[0]]
        out.append(dict(kind=kind, id=sid, files=list(fs), names=[names[f] for f in fs], proportions=list(w)))

    if "control" in kinds:
        for f in files:
            add("control", [f], [1])
    if "pair" in kinds:
        for a, b in itertools.combinations(files, 2):
            add("pair", [a, b], pair_ratio)
            add("pair", [b, a], pair_ratio)
    if "triple" in kinds:
        if len(files) < 3:
            sys.exit("A triple needs three disease files.")
        for trio in itertools.combinations(files, 3):
            for s in range(3):                                   # rotations: each disease once per role
                add("triple", list(trio[s:]) + list(trio[:s]), triple_ratio)
    return out


def point_seed(base, scenario_id, n, iteration):
    """Independent of p, r and the distance (paired comparisons)."""
    return (zlib.crc32(f"{scenario_id}|{n}|{iteration}".encode()) ^ base) & 0x7FFFFFFF


def partition_metrics(n_true):
    with open(CLUSTER_JSON, encoding="utf-8") as f:
        clusters = json.load(f)["clustering_results"]["optimal"]["clusters"]
    with open(SYNTH_JSON, encoding="utf-8") as f:
        truth = {str(p["id"]): p["subgroup"] for p in json.load(f)["patients"]}
    real = [c for c in clusters if c.get("cluster_id") != -1]
    assigned = {}
    for c in real:
        for pid in (_cluster_members(c) or []):
            if pid in truth:
                assigned[pid] = c["cluster_id"]
    t = [truth[p] for p in assigned]
    pr = list(assigned.values())
    all_t = list(truth.values())
    all_p = [assigned.get(pid, "noise") for pid in truth]
    one = n_true == 1                       # controls: ARI is undefined with a single true group
    return {"n_clusters": float(len(real)), "k_correct": float(len(real) == n_true),
            "ari": float("nan") if one or not t else adjusted_rand(t, pr),
            "ari_all": float("nan") if one else adjusted_rand(all_t, all_p),
            "coverage": len(assigned) / len(truth) if truth else float("nan")}


def run_iteration(scen, n, seed, r, p, distance, args):
    before = _paird_result_dirs()
    try:
        if CLUSTER_JSON.exists():
            CLUSTER_JSON.unlink()
        try:
            gt.generate(scen["files"], n, scen["proportions"], seed=seed, substitution=p,
                        hpo_terms=args.hpo_terms, frequency=args.frequency, label=LABEL,
                        output_json=SYNTH_JSON, verbose=True, reporting=r)
        except Exception as e:
            print(f"[!] Patient generation failed: {e}")
            traceback.print_exc()
            return _record(None)
        set_distance(distance)
        paird_args = (["-f", SYNTH_JSON, "--gene", LABEL, "--skip-evaluation", "--output-json", str(CLUSTER_JSON)]
                      + shlex.split(args.paird_args))
        if not _cluster_in_process(paird_args) or not CLUSTER_JSON.exists():
            print("[!] pairD_phenome clustering failed. Skipping iteration.")
            return _record(None)
        try:
            m = partition_metrics(len(scen["files"]))
        except Exception as e:
            print(f"[!] Could not score the clustering: {e}")
            traceback.print_exc()
            return _record(None)
        return _record(m)
    finally:
        _remove_new_result_dirs(before)


def close_call(runs, target):
    v = np.array([x["k_correct"] for x in runs], dtype=float)
    if len(v) < 2:
        return True
    se = v.std(ddof=1) / math.sqrt(len(v))
    return abs(v.mean() - target) < CLOSE_CALL_SE * se


def point_key(kind_or_row, scenario=None, r=None, p=None, distance=None, n=None):
    if isinstance(kind_or_row, dict):
        x = kind_or_row
        return (x["scenario"], float(x["reporting"]), float(x["substitution"]), x["distance"], int(x["n_patients"]))
    return (scenario, float(r), float(p), distance, int(n))


def load_points(path):
    done = {}
    if path.exists():
        for row in csv.DictReader(open(path, newline="", encoding="utf-8")):
            if row["status"] == "ok":
                done[point_key(row)] = float(row["rate"])
    return done


class Grid:
    def __init__(self, args, w_pts, f_pts, w_raw, f_raw, done):
        self.args, self.w_pts, self.f_pts, self.w_raw, self.f_raw, self.done = args, w_pts, f_pts, w_raw, f_raw, done
        self.n_points = 0

    def rate(self, scen, r, p, dist, n):
        """Rate of runs with the correct number of clusters at one N (cached)."""
        key = point_key(None, scen["id"], r, p, dist, n)
        if key in self.done:
            print(f"[=] {scen['id']} r={r:g} p={p:g} {dist} N={n}: rate {self.done[key]:.0%} (from file)")
            return self.done[key]
        a = self.args
        n_true = len(scen["files"])
        if n < 2 * n_true:
            return float("nan")
        self.n_points += 1
        label = f"{scen['kind']} {scen['id']} r={r:g} p={p:g} {dist} N={n}"
        print("\n" + "=" * 80 + f"\n[point {self.n_points}] {label}\n" + "=" * 80)
        t0 = time.perf_counter()
        axes = dict(kind=scen["kind"], scenario=scen["id"], diseases="+".join(scen["names"]),
                    proportions=fmt_ratio(scen["proportions"]), reporting=r, substitution=p,
                    distance=dist, n_patients=n)
        runs, it, extended = [], 0, False

        def one():
            nonlocal it
            it += 1
            seed = point_seed(a.base_seed, scen["id"], n, it)
            print(f"--- Iteration {it} (seed {seed}) ---")
            m = run_iteration(scen, n, seed, r, p, dist, a)
            if m:
                runs.append(m)
                print(f"    clusters {int(m['n_clusters'])} (correct: {bool(m['k_correct'])}) | ARI {m['ari']:.3f} "
                      f"| ARI all {m['ari_all']:.3f} | coverage {m['coverage']:.0%}")
                self.w_raw.writerow({**axes, "iteration": it, "seed": seed, **m})
                self.f_raw.flush()

        while it < a.iter_initial:
            one()
        if len(runs) >= 2 and close_call(runs, a.target):
            extended = True
            print(f"[*] Close call: up to {a.iter_max} iterations.")
            while it < a.iter_max:
                one()
        row = {**axes, "n_true": n_true, "iterations_run": it, "successful_iterations": len(runs),
               "close_call": extended}
        if len(runs) < MIN_SUCCESSFUL:
            row.update(status="failed", rate="", passed="")
            rate = float("nan")
        else:
            rate = float(np.mean([x["k_correct"] for x in runs]))
            row.update(status="ok", rate=rate, passed=rate >= a.target,
                       n_clusters_max=max(x["n_clusters"] for x in runs),
                       **{f"{m}_mean": float(np.nanmean([x[m] for x in runs])) for m in METRICS})
            self.done[point_key(row)] = rate
        self.w_pts.writerow(row)
        self.f_pts.flush()
        print(f"[Point] {label}: correct number of clusters in {rate:.0%} of {len(runs)} runs "
              f"({time.perf_counter() - t0:.0f} s)")
        return rate

    def search(self, scen, r, p, dist):
        """Smallest N on the ladder with rate >= target. Returns (n_needed or None, {N: rate})."""
        a = self.args
        ladder = [n for n in a.ladder if n >= 2 * len(scen["files"])]
        seen = {}
        if a.full:
            for n in ladder:
                seen[n] = self.rate(scen, r, p, dist, n)
            ok = [n for n in ladder if np.isfinite(seen[n]) and seen[n] >= a.target]
            return (ok[0] if ok else None), seen
        lo, hi = -1, len(ladder)            # invariant: ladder[lo] fails, ladder[hi] passes
        # first rung: --start (or the nearest rung above it); then plain bisection
        first = next((i for i, n in enumerate(ladder) if n >= a.start), len(ladder) // 2)
        while hi - lo > 1:
            mid = first if first is not None else (lo + hi) // 2
            first = None
            seen[ladder[mid]] = v = self.rate(scen, r, p, dist, ladder[mid])
            if np.isfinite(v) and v >= a.target:
                hi = mid
            else:
                lo = mid
        return (ladder[hi] if hi < len(ladder) else None), seen


def parse_ratio(text):
    return [float(x) for x in text.split(":")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diseases", nargs="+", default=DISEASE_FILES, help="Orphanet symptom lists (.txt or .csv)")
    ap.add_argument("--kinds", nargs="+", default=KINDS, choices=KINDS)
    ap.add_argument("--pair-ratio", type=parse_ratio, default=[2, 1])
    ap.add_argument("--triple-ratio", type=parse_ratio, default=[3, 2, 1])
    ap.add_argument("--ladder", nargs="+", type=int, default=LADDER, help="cohort sizes searched")
    ap.add_argument("--start", type=int, default=START_N, help="first N tried by the bisection")
    ap.add_argument("--control-n", nargs="+", type=int, default=CONTROL_N,
                    help="sizes of the controls (default 10 30 90 180)")
    ap.add_argument("--target", type=float, default=0.8, help="rate of correct runs required")
    ap.add_argument("--full", action="store_true", help="evaluate every rung instead of bisecting")
    ap.add_argument("--substitution", nargs="+", type=float, default=SUBSTITUTION)
    ap.add_argument("--reporting", nargs="+", type=float, default=REPORTING)
    ap.add_argument("--distance", nargs="+", default=DISTANCES, choices=DISTANCE_CHOICES)
    ap.add_argument("--frequency", choices=["sample", "midpoint", "fixed"], default="sample")
    ap.add_argument("--hpo-terms", default="hpo_terms.json")
    ap.add_argument("--base-seed", type=int, default=1)
    ap.add_argument("--iter-initial", type=int, default=N_ITER_INITIAL)
    ap.add_argument("--iter-max", type=int, default=N_ITER_MAX)
    ap.add_argument("--organ-mode", choices=["nearest", "penalty"], default="nearest")
    ap.add_argument("-o", "--output", default="test2_points.csv")
    ap.add_argument("--raw", default="test2_iterations.csv")
    ap.add_argument("--searches", default="test2_n_needed.csv")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--keep-results", action="store_true")
    ap.add_argument("--paird-args", default="",
                    help='Extra options passed to pairD_phenome.py, e.g. "--gridsearch 3 --min-dist 0". '
                         'Use a separate folder (or -o/--raw/--searches names) per setting: they are not part of the --resume key.')
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args.ladder = sorted(set(args.ladder))
    args.control_n = sorted(set(args.control_n))

    for f in args.diseases:
        if not Path(f).exists():
            sys.exit(f"{f} not found.")
    if not Path(args.hpo_terms).exists():
        sys.exit(f"{args.hpo_terms} not found.")
    scenarios = build_scenarios(args.diseases, args.kinds, args.pair_ratio, args.triple_ratio)
    for f in args.diseases:
        rows = gt.load_disease(f)
        cls = sum(c is not None for _, _, c, _ in rows)
        print(f"{gt.disease_name(f)}: {len(rows)} symptoms ({cls} with an Orphanet class)")
    combos = [(r, p, d) for r in args.reporting for p in args.substitution for d in args.distance]
    n_ctrl = sum(s["kind"] == "control" for s in scenarios)
    n_search = len(scenarios) - n_ctrl
    per_search = len(args.ladder) if args.full else math.ceil(math.log2(len(args.ladder) + 1))
    if args.paird_args:
        print(f"pairD_phenome.py extra options: {args.paird_args}")
    print(f"\n=== TEST 2: {len(scenarios)} scenarios ({n_ctrl} controls, {n_search} pairs/triples) x "
          f"{len(combos)} (r, p, distance) combinations ===")
    for s in scenarios:
        print(f"  {s['kind']:<8}{s['id']}")
    print(f"Ladder {args.ladder} (search starts at N = {args.start}), target {args.target:.0%}, controls at N = {args.control_n}")
    print(f"About {n_ctrl * len(combos) * len(args.control_n) + n_search * len(combos) * per_search} "
          f"points of {args.iter_initial}-{args.iter_max} iterations.")
    if args.dry_run:
        return

    os.environ["PGM_ORGAN_MODE"] = args.organ_mode
    try:
        mp.set_start_method("fork", force=True)
    except (RuntimeError, ValueError) as e:
        print(f"[!] Could not set multiprocessing start method to 'fork': {e}")
    global KEEP_RESULTS
    KEEP_RESULTS = args.keep_results
    for d in args.distance:
        set_distance(d)                                  # checks pgmodules.py (and phenotype.hpoa) now

    out, raw, srch = Path(args.output), Path(args.raw), Path(args.searches)
    done = load_points(out) if args.resume else {}
    t0 = time.perf_counter()
    with open(out, "a" if args.resume and out.exists() else "w", newline="", encoding="utf-8") as f_pts, \
            open(raw, "a" if args.resume and raw.exists() else "w", newline="", encoding="utf-8") as f_raw:
        w_pts = csv.DictWriter(f_pts, fieldnames=POINT_FIELDS, extrasaction="ignore")
        w_raw = csv.DictWriter(f_raw, fieldnames=RAW_FIELDS, extrasaction="ignore")
        if f_pts.tell() == 0:
            w_pts.writeheader()
        if f_raw.tell() == 0:
            w_raw.writeheader()
        grid = Grid(args, w_pts, f_pts, w_raw, f_raw, done)
        searches = []
        for scen in scenarios:
            for r, p, dist in combos:
                if scen["kind"] == "control":
                    for n in args.control_n:
                        grid.rate(scen, r, p, dist, n)
                    continue
                n_needed, seen = grid.search(scen, r, p, dist)
                searches.append(dict(kind=scen["kind"], scenario=scen["id"], diseases="+".join(scen["names"]),
                                     proportions=fmt_ratio(scen["proportions"]), reporting=r, substitution=p,
                                     distance=dist, target=args.target,
                                     n_needed="" if n_needed is None else n_needed, reached=n_needed is not None,
                                     ladder_max=max(args.ladder),
                                     evaluated=";".join(f"{n}:{v:.2f}" for n, v in sorted(seen.items()))))
                print(f"[Search] {scen['id']} r={r:g} p={p:g} {dist}: N needed "
                      f"{n_needed if n_needed else '> ' + str(max(args.ladder))}")
                # rewritten after every search, so it is complete whenever the run stops
                with open(srch, "w", newline="", encoding="utf-8") as f_s:
                    w = csv.DictWriter(f_s, fieldnames=SEARCH_FIELDS)
                    w.writeheader()
                    w.writerows(searches)
    print(f"\n[Complete] TEST 2 finished in {time.perf_counter() - t0:.0f} s. "
          f"Results: {out} (per N), {raw} (per iteration), {srch} (N needed).")


if __name__ == "__main__":
    main()
