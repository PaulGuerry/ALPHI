#!/usr/local/bin/python3.14
"""
Compares pairD_phenome.py clusterings of the same real cohort (e.g. the MYO5B patients),
run with different distances (OB legacy/nearest, Jaccard, Resnik), by the SIGNATURE
FEATURES of the clusters found: does each run find clusters whose features match the
expected phenotypes (e.g. MVID, PFIC, MVID + PFIC)? No patient label is used as truth.

Expected phenotypes (--expected, JSON; a template is written by --write-template):
  {"MVID": {"present": ["HP:0002014", ...], "absent": ["HP:0001396", ...]}, ...}
  present  features expected in most patients of such a cluster
  absent   features expected to be rare in it
  A feature counts as present in a patient when the patient has the term or any of its
  descendants (e.g. HP:0002014 Diarrhea covers chronic, intractable, secretory diarrhea),
  using hpo_terms.json.

For each run (the optimal clustering in its --output-json file):
  - each cluster's size and the prevalence of every expected feature (cluster vs rest)
  - match score of each cluster with each expected phenotype:
        mean over the phenotype's features of  prevalence (present)  or  1 - prevalence (absent)
    1 = every patient has all the expected features and none of the excluded ones
  - the best-matching phenotype of each cluster, and whether the expected phenotypes are
    found one-to-one (each matched by exactly one cluster)
  - pairD_phenome.py's own associated symptoms of each cluster (top --top), with names
Between runs: agreement of the clusterings (ARI on the patients both assigned).

Articles (--articles, needs --patients): is a run's clustering driven by the source
publication (e.g. reporting vocabulary) rather than by phenotype? Per run:
  - a table of articles (with at least --min-article patients in the cohort) x clusters
  - article concentration: share of patients in their article's majority cluster, compared
    with the same share after shuffling the cluster labels (--permutations times): a value
    far above the shuffled one means clusters follow articles. Given for all clusters and
    for each PAIR of clusters (the informative test: e.g. two MVID-like clusters that follow
    the articles reflect reporting, not phenotype)
  - ARI between clusters and articles
--assignments writes one row per patient: id, article, cluster in each run.

Patients' symptoms are read from the cluster file (what pairD_phenome.py used); with
--patients, from the patient database instead (by id).

Usage (in ~/myo5b_distances):
  python3.14 compare_myo5b_distances.py --write-template expected_myo5b.json   # then edit it
  python3.14 compare_myo5b_distances.py --expected expected_myo5b.json --hpo-terms hpo_terms.json \
      OB_legacy=ob_legacy/clusters.json OB=ob_nearest/clusters.json \
      Jaccard=jaccard/clusters.json Resnik=resnik/clusters.json -o myo5b_clusters.csv \
      --articles --patients ~/GITHUB/pairD_phenome/261013_patientData.json --assignments myo5b_patients.csv
"""
import argparse
import csv
import itertools
import json
import sys
from collections import Counter
from math import comb

TEMPLATE = {
    "_comment": "Edit freely. present = expected in most patients of the cluster; absent = expected to "
                "be rare. Terms match their descendants. Names are for reading only.",
    "MVID": {
        "present": ["HP:0002014", "HP:0033995", "HP:0033994"],
        "absent": ["HP:0001396", "HP:0000989"],
    },
    "PFIC": {
        "present": ["HP:0001396", "HP:0000989"],
        "absent": ["HP:0002014", "HP:0033995", "HP:0033994"],
    },
    "MVID+PFIC": {
        "present": ["HP:0002014", "HP:0033995", "HP:0033994", "HP:0001396", "HP:0000989"],
        "absent": [],
    },
}


def adjusted_rand(true, pred):
    n = len(true)
    if n < 2:
        return float("nan")
    cont = Counter(zip(true, pred))
    a, b = Counter(true), Counter(pred)
    sum_ij = sum(comb(v, 2) for v in cont.values())
    sum_a = sum(comb(v, 2) for v in a.values())
    sum_b = sum(comb(v, 2) for v in b.values())
    expected = sum_a * sum_b / comb(n, 2)
    max_index = (sum_a + sum_b) / 2
    if max_index == expected:
        return 1.0
    return (sum_ij - expected) / (max_index - expected)


def code(x):
    """'0001396', 'HP:0001396', ['0001396', None, None] -> 'HP:0001396'."""
    if isinstance(x, (list, tuple)):
        x = x[0]
    if isinstance(x, dict):
        x = x.get("hpo_code") or x.get("code") or x.get("id")
    x = str(x).strip()
    return x if x.startswith("HP:") else "HP:" + x.zfill(7)


class HPO:
    def __init__(self, path):
        d = json.load(open(path, encoding="utf-8"))
        self.terms, self.alt, self._anc = d["terms"], d.get("alt_id_map", {}), {}

    def name(self, t):
        return self.terms.get(self.alt.get(t, t), {}).get("name", t)

    def up(self, t):
        t = self.alt.get(t, t)
        hit = self._anc.get(t)
        if hit is None:
            hit = {t}
            for p in self.terms.get(t, {}).get("parents", []):
                hit |= self.up(p)
            self._anc[t] = hit = frozenset(hit)
        return hit


def load_run(path, solution="optimal"):
    """{patient id: cluster ('C0', ... or 'noise')}, {id: [codes]}, {cluster: associated symptoms}"""
    d = json.load(open(path, encoding="utf-8"))
    res = d["clustering_results"]
    if solution not in res:
        sys.exit(f"{path}: no '{solution}' solution (available: {', '.join(res)}).")
    assign, symptoms, assoc = {}, {}, {}
    for c in res[solution]["clusters"]:
        cid = c.get("cluster_id")
        name = "noise" if cid == -1 else f"C{cid}"
        assoc[name] = c.get("associated_symptoms") or []
        for p in c.get("patients") or []:
            pid = str(p.get("id") if isinstance(p, dict) else p)
            assign[pid] = name
            if isinstance(p, dict) and p.get("symptoms"):
                symptoms[pid] = [code(s) for s in p["symptoms"]]
    return assign, symptoms, assoc


def concentration(pids, assign, article_of):
    """Share of patients in the majority cluster of their article."""
    by_art = {}
    for p in pids:
        by_art.setdefault(article_of.get(p, "unknown"), []).append(assign[p])
    return sum(Counter(v).most_common(1)[0][1] for v in by_art.values()) / len(pids)


def article_report(label, assign, clusters, article_of, args):
    import random
    pids = [p for p in assign if assign[p] != "noise"]
    if not pids:
        return
    arts = Counter(article_of.get(p, "unknown") for p in pids)
    print(f"\n  Articles: {len(arts)} articles for {len(pids)} assigned patients")
    shown = [a for a, n in arts.most_common() if n >= args.min_article]
    width = max([len(a) for a in shown] + [10])
    width = min(width, 48)
    print(f"    {'article':<{width}}" + "".join(f"{c:>6}" for c in clusters) + f"{'n':>6}  majority")
    for a in shown:
        cnt = Counter(assign[p] for p in pids if article_of.get(p, "unknown") == a)
        maj, k = cnt.most_common(1)[0]
        print(f"    {a[:width]:<{width}}" + "".join(f"{cnt.get(c, 0):>6}" for c in clusters)
              + f"{arts[a]:>6}  {maj} {k / arts[a]:.0%}")
    if len(shown) < len(arts):
        print(f"    ({len(arts) - len(shown)} articles with fewer than {args.min_article} patients not shown)")
    rng = random.Random(1)

    def test(sub):
        obs = concentration(sub, assign, article_of)
        labels = [assign[p] for p in sub]
        null = []
        for _ in range(args.permutations):
            rng.shuffle(labels)
            null.append(concentration(sub, dict(zip(sub, labels)), article_of))
        null.sort()
        return obs, null[len(null) // 2], sum(x >= obs for x in null)

    obs, med, exceed = test(pids)
    ari = adjusted_rand([article_of.get(p, "unknown") for p in pids], [assign[p] for p in pids])
    print(f"    all clusters: article concentration {obs:.0%} (shuffled: {med:.0%}; {exceed}/{args.permutations} "
          f"shuffles >= observed) | ARI clusters vs articles {ari:.3f}")
    print("    (articles often report one phenotype, so clusters that differ in phenotype also differ in"
          " articles: the pairwise test below is the informative one)")
    for c1, c2 in itertools.combinations(clusters, 2):
        sub = [p for p in pids if assign[p] in (c1, c2)]
        shared = {article_of.get(p, "unknown") for p in sub if assign[p] == c1} & \
                 {article_of.get(p, "unknown") for p in sub if assign[p] == c2}
        obs, med, exceed = test(sub)
        print(f"    {c1} vs {c2}: {len(shared)} articles contribute to both | article concentration {obs:.0%} "
              f"(shuffled {med:.0%}; {exceed}/{args.permutations} shuffles >= observed)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="LABEL=path/to/clusters.json")
    ap.add_argument("--expected", help="expected phenotypes (JSON)")
    ap.add_argument("--write-template", metavar="PATH", help="write an expected-phenotypes template and stop")
    ap.add_argument("--hpo-terms", default="hpo_terms.json")
    ap.add_argument("--patients", default=None, help="patient database JSON (symptoms by id), optional")
    ap.add_argument("--top", type=int, default=8, help="associated symptoms shown per cluster")
    ap.add_argument("--solution", default="optimal", choices=["optimal", "second", "third"],
                    help="which of pairD_phenome.py's ranked clusterings to compare (default optimal)")
    ap.add_argument("-o", "--output", default=None, help="CSV: one row per run x cluster")
    ap.add_argument("--articles", action="store_true", help="cluster x article analysis (needs --patients)")
    ap.add_argument("--min-article", type=int, default=3, help="articles shown in the table: at least this many patients")
    ap.add_argument("--permutations", type=int, default=1000)
    ap.add_argument("--assignments", default=None, help="CSV: one row per patient (id, article, clusters)")
    args = ap.parse_args()

    if args.write_template:
        json.dump(TEMPLATE, open(args.write_template, "w", encoding="utf-8"), indent=2)
        print(f"Wrote {args.write_template}: edit the expected features, then rerun with --expected.")
        return
    if not args.runs or not args.expected:
        sys.exit("Give --expected and at least one LABEL=clusters.json (or --write-template).")

    H = HPO(args.hpo_terms)
    expected = {k: v for k, v in json.load(open(args.expected, encoding="utf-8")).items()
                if not k.startswith("_")}
    features = []
    for ph in expected.values():
        for f in ph.get("present", []) + ph.get("absent", []):
            if f not in features:
                features.append(f)
    print("Expected phenotypes:")
    for ph, v in expected.items():
        print(f"  {ph}: present " + ", ".join(H.name(f) for f in v.get("present", []))
              + ("; absent " + ", ".join(H.name(f) for f in v.get("absent", [])) if v.get("absent") else ""))

    if args.articles and not args.patients:
        sys.exit("--articles needs --patients (the articles are read from the patient database).")
    db_sym, article_of = {}, {}
    if args.patients:
        db = json.load(open(args.patients, encoding="utf-8"))
        db = db if isinstance(db, list) else next(v for v in db.values() if isinstance(v, list))
        db_sym = {str(p["id"]): [code(s) for s in p.get("symptoms") or []] for p in db}
        for p in db:
            arts = [a for a in p.get("articles") or [] if isinstance(a, dict)]
            if arts:
                doi = str(arts[0].get("doi") or "").replace("https://doi.org/", "").replace("http://dx.doi.org/", "")
                article_of[str(p["id"])] = f"{arts[0].get('publication_year', '')} {doi}".strip() or "unknown"
            else:
                article_of[str(p["id"])] = str(p.get("pmid") or "unknown")

    runs, rows = {}, []
    for item in args.runs:
        if "=" not in item:
            sys.exit(f"Give runs as LABEL=path, not {item!r}")
        label, path = item.split("=", 1)
        assign, sym, assoc = load_run(path, args.solution)
        if db_sym:
            sym = {pid: db_sym.get(pid, sym.get(pid, [])) for pid in assign}
        missing = [pid for pid in assign if pid not in sym]
        if missing:
            sys.exit(f"{label}: no symptoms for {len(missing)} patients (e.g. {missing[0]}); give --patients.")
        runs[label] = assign
        has = {pid: {f: any(f in H.up(t) for t in sym[pid]) for f in features} for pid in assign}
        clusters = sorted({c for c in assign.values() if c != "noise"}, key=lambda x: int(x[1:]))
        n_all = len(assign)
        n_ass = sum(c != "noise" for c in assign.values())
        print(f"\n{'=' * 78}\n{label} ({args.solution} solution): {len(clusters)} clusters, {n_ass}/{n_all} patients assigned "
              f"({n_ass / n_all:.0%})\n{'=' * 78}")

        best_of = {}
        for cl in clusters + (["noise"] if n_ass < n_all else []):
            mem = [p for p in assign if assign[p] == cl]
            rest = [p for p in assign if assign[p] != cl and assign[p] != "noise"]
            prev = {f: sum(has[p][f] for p in mem) / len(mem) for f in features}
            prev_rest = {f: (sum(has[p][f] for p in rest) / len(rest)) if rest else float("nan") for f in features}
            print(f"\n  {cl} ({len(mem)} patients)")
            print(f"    {'feature':<42}{'cluster':>9}{'rest':>7}")
            for f in features:
                print(f"    {H.name(f)[:41]:<42}{prev[f]:>9.0%}{prev_rest[f]:>7.0%}")
            if cl == "noise":
                continue
            scores = {}
            for ph, v in expected.items():
                parts = [prev[f] for f in v.get("present", [])] + [1 - prev[f] for f in v.get("absent", [])]
                scores[ph] = sum(parts) / len(parts) if parts else float("nan")
            best = max(scores, key=scores.get)
            best_of[cl] = best
            print("    match: " + " | ".join(f"{ph} {s:.2f}" for ph, s in scores.items())
                  + f"  ->  {best}")
            top = assoc.get(cl, [])[: args.top]
            if top:
                print("    pairD associated symptoms (cluster vs rest): " + "; ".join(
                    f"{H.name(code(a.get('hpo_code')))} {a.get('cluster_prevalence', '?')} vs "
                    f"{a.get('rest_cohort_prevalence', '?')}" for a in top))
            rows.append({"run": label, "cluster": cl, "n": len(mem), "best_match": best,
                         **{f"score_{ph}": round(s, 3) for ph, s in scores.items()},
                         **{f"prev_{f}": round(prev[f], 3) for f in features}})

        found = Counter(best_of.values())
        one_to_one = all(found.get(ph, 0) == 1 for ph in expected) and len(best_of) == len(expected)
        print(f"\n  Expected phenotypes found: " + ", ".join(f"{ph} x{found.get(ph, 0)}" for ph in expected)
              + f"  ->  {'one-to-one' if one_to_one else 'NOT one-to-one'}")
        for r in rows:
            if r["run"] == label:
                r["one_to_one"] = one_to_one
        if args.articles:
            article_report(label, assign, clusters, article_of, args)

    if len(runs) > 1:
        print(f"\n{'=' * 78}\nAgreement between runs (ARI on patients assigned in both)")
        for a, b in itertools.combinations(runs, 2):
            both = [p for p in runs[a] if runs[a][p] != "noise" and runs[b].get(p, "noise") != "noise"]
            v = adjusted_rand([runs[a][p] for p in both], [runs[b][p] for p in both]) if both else float("nan")
            print(f"  {a} vs {b}: {v:.3f} ({len(both)} patients)")

    if args.assignments:
        ids = list(dict.fromkeys(pid for a in runs.values() for pid in a))
        with open(args.assignments, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "article"] + list(runs))
            for pid in ids:
                w.writerow([pid, article_of.get(pid, "")] + [runs[r].get(pid, "not in run") for r in runs])
        print(f"\nPer-patient assignments: {args.assignments}")

    if args.output and rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nPer-cluster table: {args.output}")


if __name__ == "__main__":
    main()
