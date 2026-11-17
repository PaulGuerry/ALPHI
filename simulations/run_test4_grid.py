#!/usr/local/bin/python3.14
"""
Simulation 4 grid: cohorts with a combined subgroup and a marker with two causes
(generate_test4_patients.py), clustered with pairD_phenome.py (ALPHI), one fresh process per run.

Axes (each restricted on the command line):
  --f            frequency of the secondary marker in subgroup A (0 = no marker with two causes)
  --ab-specific  symptoms present in the combined subgroup only (0 = AB is exactly A + B)
  --reporting    r; --substitution p; --k signature size; --sizes A B AB; --distance
The generation seed does not depend on f, p or the distance, so these comparisons are made on
the same underlying patients.

Per run: number of clusters, k correct (3), ARI (assigned patients) and ARI with unassigned
patients as a group of their own, coverage, whether the combined subgroup has a cluster of its
own, and, for the patients of A that carry the secondary marker, the share placed in the main
cluster of A (sec_in_A) and in the main cluster of AB (sec_in_AB), with the same shares for the
other patients of A for comparison.

Each cohort is also saved in cohorts/<setting>_c<iteration>.json for build_llm_test4.py.

Usage (in a folder of its own; pairD_phenome.py on PATH):
  python3.14 run_test4_grid.py -t test1_table_C1.csv --hpo-terms hpo_terms.json \
      --f 0 0.3 --ab-specific 0 3 --reporting 1 0.75 --substitution 0.1 \
      --distance dphenome jaccard resnik -o test4_iterations.csv 2>&1 | tee test4.log
  ... --paird-args="--select-terms"    (extra options for pairD_phenome.py)
Simulation 5 (meaningful terms; the marker is Cholestasis, rank 1 of B; partial B in AB):
  python3.14 run_test4_grid.py -t test5_table_myo5b.csv --hpo-terms hpo_terms.json --prefix s5 \
      --secondary 1 --f 0.3 --ab-specific 0 3 --ab-b 1 0.4 --reporting 1 --substitution 0.1 \
      --distance dphenome --paird-args="--lca" -o test5_lca_iterations.csv
"""
import argparse
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
import zlib
from collections import Counter
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_test4_patients as g4                       # noqa: E402

SYNTH_JSON = "synthetic_patients.json"
CLUSTER_JSON = Path("cluster_visualization_current.json")
FIELDS = ["setting", "iteration", "seed", "sizes", "sig_size", "f", "ab_specific", "ab_b", "reporting", "substitution",
          "distance", "n_secondary", "n_clusters", "k_correct", "ari", "ari_all", "coverage", "ab_own_cluster",
          "sec_in_A", "sec_in_AB", "other_A_in_A", "other_A_in_AB"]


def tag(x):
    return f"{x:g}".replace(".", "")


def setting_name(sizes, k, f, ab, r, p, ab_b=1.0, prefix="s4"):
    b = f"_b{tag(ab_b)}" if ab_b != 1 else ""
    return f"{prefix}_N{sum(sizes)}_s{k}_p{tag(p)}_f{tag(f)}_ab{ab}{b}_r{tag(r)}"


def cohort_seed(base, sizes, k, ab, r, it):
    """Independent of f, p and the distance (paired comparisons)."""
    return (zlib.crc32(f"test4|{sizes}|{k}|{ab}|{r:g}|{it}".encode()) ^ base) & 0x7FFFFFFF


def ari(a, b):
    n = len(a)
    if n < 2:
        return float("nan")
    cont, ra, rb = Counter(zip(a, b)), Counter(a), Counter(b)
    s = sum(comb(v, 2) for v in cont.values())
    sa, sb = sum(comb(v, 2) for v in ra.values()), sum(comb(v, 2) for v in rb.values())
    exp = sa * sb / comb(n, 2)
    mx = (sa + sb) / 2
    return 1.0 if mx == exp else (s - exp) / (mx - exp)


def members(cluster):
    for key in ("patients", "patient_ids", "members", "ids"):
        val = cluster.get(key)
        if isinstance(val, list) and val:
            return [str(p.get("id") or p.get("patient_id")) if isinstance(p, dict) else str(p) for p in val]
    return []


def score(patients):
    clusters = json.load(open(CLUSTER_JSON, encoding="utf-8"))["clustering_results"]["optimal"]["clusters"]
    truth = {p["id"]: p["subgroup"] for p in patients}
    sec = {p["id"] for p in patients if p["secondary"]}
    assigned = {}
    for c in clusters:
        if c.get("cluster_id") == -1:
            continue
        for pid in members(c):
            if pid in truth:
                assigned[pid] = c["cluster_id"]
    ids = list(assigned)
    k = len({assigned[i] for i in ids})
    main = {}
    for g in (1, 2, 3):
        cnt = Counter(assigned[i] for i in ids if truth[i] == g)
        main[g] = cnt.most_common(1)[0][0] if cnt else None
    a_ids = [p["id"] for p in patients if p["subgroup"] == 1]

    def share(group, cluster):
        g = [i for i in group if i in assigned]
        return sum(assigned[i] == cluster for i in g) / len(g) if g and cluster is not None else float("nan")

    ab_own = main[3] is not None and main[3] not in (main[1], main[2])
    sec_a = [i for i in a_ids if i in sec]
    oth_a = [i for i in a_ids if i not in sec]
    return dict(n_clusters=k, k_correct=int(k == 3), ari=ari([truth[i] for i in ids], [assigned[i] for i in ids]),
                ari_all=ari(list(truth.values()), [assigned.get(i, "noise") for i in truth]),
                coverage=len(ids) / len(truth), ab_own_cluster=int(ab_own),
                sec_in_A=share(sec_a, main[1]), sec_in_AB=share(sec_a, main[3]) if ab_own else float("nan"),
                other_A_in_A=share(oth_a, main[1]),
                other_A_in_AB=share(oth_a, main[3]) if ab_own else float("nan"))


def run_paird(distance, args):
    exe = shutil.which("pairD_phenome.py")
    if not exe:
        sys.exit("pairD_phenome.py not found on PATH.")
    env = dict(os.environ, PGM_DISTANCE=distance)
    cmd = [sys.executable, os.path.realpath(exe), "-f", SYNTH_JSON, "--gene", g4.LABEL, "--skip-evaluation",
           "--output-json", str(CLUSTER_JSON), "--min-symptoms", "1"] + shlex.split(args.paird_args)
    return subprocess.run(cmd, env=env).returncode == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-t", "--table", required=True)
    ap.add_argument("--hpo-terms", default="hpo_terms.json")
    ap.add_argument("--sizes", nargs=3, type=int, default=[70, 45, 25])
    ap.add_argument("--k", nargs="+", type=int, default=[6])
    ap.add_argument("--f", nargs="+", type=float, default=[0.0, 0.3])
    ap.add_argument("--ab-specific", nargs="+", type=int, default=[0, 3])
    ap.add_argument("--ab-b", nargs="+", type=float, default=[1.0],
                    help="presence of B's symptoms in AB relative to B (generate_test4_patients.py --ab-b)")
    ap.add_argument("--secondary", nargs="+", type=int, default=[4], help="ranks of set B used as the marker")
    ap.add_argument("--prefix", default="s4", help="setting-name prefix (s5 for Simulation 5)")
    ap.add_argument("--reporting", nargs="+", type=float, default=[1.0, 0.75])
    ap.add_argument("--substitution", nargs="+", type=float, default=[0.1])
    ap.add_argument("--distance", nargs="+", default=["dphenome", "jaccard", "resnik"])
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--base-seed", type=int, default=4)
    ap.add_argument("--paird-args", default="", help='extra options for pairD_phenome.py, e.g. "--select-terms"')
    ap.add_argument("--keep-results", action="store_true", help="keep pairD_phenome.py's run folders")
    ap.add_argument("-o", "--output", default="test4_iterations.csv")
    ap.add_argument("--resume", action="store_true", help="skip runs already in the output file")
    a = ap.parse_args()

    out = Path(a.output)
    done = set()
    if a.resume and out.exists():
        done = {(r["setting"], r["iteration"], r["distance"]) for r in csv.DictReader(open(out, encoding="utf-8"))}
    new_file = not out.exists() or not a.resume
    fh = open(out, "w" if new_file else "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
    if new_file:
        w.writeheader()
    Path("cohorts").mkdir(exist_ok=True)
    points = [(k, f, ab, r, p, bb) for k in a.k for ab in a.ab_specific for r in a.reporting
              for p in a.substitution for bb in a.ab_b for f in a.f]
    print(f"=== Simulation 4: {len(points)} settings x {a.iterations} cohorts x {len(a.distance)} distances"
          + (f"; pairD_phenome.py options: {a.paird_args}" if a.paird_args else ""))
    for k, f, ab, r, p, bb in points:
        name = setting_name(a.sizes, k, f, ab, r, p, bb, a.prefix)
        for it in range(1, a.iterations + 1):
            seed = cohort_seed(a.base_seed, a.sizes, k, ab, r, it)
            meta = g4.generate(a.table, a.sizes, k, f=f, ab_specific=ab, reporting=r, substitution=p,
                               hpo_terms=a.hpo_terms, seed=seed, output_json=SYNTH_JSON, verbose=False,
                               secondary=tuple(a.secondary), ab_b=bb)
            shutil.copy(SYNTH_JSON, Path("cohorts") / f"{name}_c{it}.json")
            patients = json.load(open(SYNTH_JSON, encoding="utf-8"))["patients"]
            for dist in a.distance:
                if (name, str(it), dist) in done:
                    continue
                print(f"--- {name} c{it} {dist} (seed {seed}; {meta['n_secondary']} A patients with the marker)",
                      flush=True)
                before = {d for d in Path(".").iterdir() if d.is_dir() and d.name[:6].isdigit()}
                if CLUSTER_JSON.exists():
                    CLUSTER_JSON.unlink()
                row = dict(setting=name, iteration=it, seed=seed, sizes="/".join(map(str, a.sizes)), sig_size=k, f=f,
                           ab_specific=ab, ab_b=bb, reporting=r, substitution=p, distance=dist,
                           n_secondary=meta["n_secondary"])
                if run_paird(dist, a) and CLUSTER_JSON.exists():
                    row.update(score(patients))
                    print(f"    clusters {row['n_clusters']} | ARI {row['ari']:.2f} | AB own cluster "
                          f"{row['ab_own_cluster']} | marker carriers in A {row['sec_in_A']:.2f}, in AB "
                          f"{row['sec_in_AB']:.2f} | other A in A {row['other_A_in_A']:.2f}", flush=True)
                else:
                    print("    [!] clustering failed")
                w.writerow(row)
                fh.flush()
                if not a.keep_results:
                    for d in {d for d in Path(".").iterdir() if d.is_dir() and d.name[:6].isdigit()} - before:
                        shutil.rmtree(d, ignore_errors=True)
    fh.close()
    print(f"[Complete] Results: {out}; cohorts in cohorts/")


if __name__ == "__main__":
    main()
