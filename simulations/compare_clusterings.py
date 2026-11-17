#!/usr/local/bin/python3.14
"""
Compares two ALPHI (pairD_phenome.py) clusterings of overlapping cohorts, e.g. the main
MYO5B analysis and the sensitivity analysis restricted to genotyped patients: number of
clusters, cross-table of the patients present in both, adjusted Rand index (ARI) on those
patients, and cluster sizes and stabilities of each solution.

Usage:
  python3.14 compare_clusterings.py clusters.json clusters_genotyped.json \
      --names full genotyped [--solution optimal]
"""
import argparse
import json
from math import comb


def load(path, solution):
    sol = json.load(open(path, encoding="utf-8"))["clustering_results"][solution]
    out, info = {}, []
    for c in sol["clusters"]:
        cid = c.get("cluster_id")
        name = "noise" if cid == -1 else f"{cid}:{c.get('cluster_name') or cid}"
        ids = [str(p["id"]) if isinstance(p, dict) else str(p) for p in (c.get("patients") or [])]
        for i in ids:
            out[i] = name
        if cid != -1:
            info.append((name, len(ids), c.get("stability_jaccard_mean"), c.get("flag_few_features")))
    return out, info, sol.get("num_clusters_k")


def ari(a, b):
    n = len(a)
    cont, ra, rb = {}, {}, {}
    for x, y in zip(a, b):
        cont[(x, y)] = cont.get((x, y), 0) + 1
        ra[x] = ra.get(x, 0) + 1
        rb[y] = rb.get(y, 0) + 1
    s = sum(comb(v, 2) for v in cont.values())
    sa, sb = sum(comb(v, 2) for v in ra.values()), sum(comb(v, 2) for v in rb.values())
    exp = sa * sb / comb(n, 2)
    mx = (sa + sb) / 2
    return 1.0 if mx == exp else (s - exp) / (mx - exp)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--names", nargs=2, default=["A", "B"])
    ap.add_argument("--solution", default="optimal", help="optimal or second")
    x = ap.parse_args()

    A, ia, ka = load(x.a, x.solution)
    B, ib, kb = load(x.b, x.solution)
    for nm, info, k, d in [(x.names[0], ia, ka, A), (x.names[1], ib, kb, B)]:
        print(f"{nm}: k = {k}, {len(d)} patients")
        for name, n, st, fl in info:
            print(f"   {name:<28} n = {n:<4} stability {st if st is None else round(st, 2)}"
                  + ("  FLAGGED" if fl else ""))
    shared = sorted(set(A) & set(B))
    print(f"\nPatients in both: {len(shared)} (only in {x.names[0]}: {len(set(A) - set(B))}, "
          f"only in {x.names[1]}: {len(set(B) - set(A))})")
    rows = sorted({A[i] for i in shared})
    cols = sorted({B[i] for i in shared})
    w = max(len(r) for r in rows + [x.names[0]]) + 2
    print(f"\n{x.names[0] + ' \\ ' + x.names[1]:<{w}}" + "".join(f"{c:>22}" for c in cols))
    for r in rows:
        print(f"{r:<{w}}" + "".join(f"{sum(A[i] == r and B[i] == c for i in shared):>22}" for c in cols))
    both = [i for i in shared if A[i] != "noise" and B[i] != "noise"]
    print(f"\nARI on the {len(both)} patients clustered in both: {ari([A[i] for i in both], [B[i] for i in both]):.3f}")


if __name__ == "__main__":
    main()
