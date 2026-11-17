#!/usr/local/bin/python3.14
"""
Why do patients with both a PFIC-type liver phenotype and an MVID-type intestinal phenotype
not form their own cluster? Uses the distances pairD_phenome.py actually computed (cdm.csv)
to test whether such "mixed" patients form a separate group in distance space.

Groups are defined by FEATURES, not by labels (--features JSON; template: --write-template):
  liver score  fraction of the liver signature features present (default: cholestasis,
               pruritus, jaundice, hyperbilirubinaemia, raised serum bile acids)
  gut score    fraction of the intestinal signature features present (default: diarrhoea,
               microvillus inclusions, dependency on parenteral nutrition, villous atrophy)
  A feature is present when the patient has the term or a descendant.
  mixed        >= --min-liver liver features AND >= --min-gut gut features
  liver only   >= --min-liver liver features, fewer than --min-gut gut features
  gut only     >= --min-gut gut features, fewer than --min-liver liver features
  other        neither

For each run (LABEL=run folder containing clusters.json and a result folder with cdm.csv):
  1. groups x clusters: where the mixed patients end up
  2. mean distance between and within groups
  3. for each mixed patient: mean distance to the other mixed patients (a) and to the
     liver-only patients (b); silhouette (b - a) / max(a, b) > 0 means closer to the other
     mixed patients. Summary: share of mixed patients with silhouette > 0, median silhouette
  4. nearest neighbours: share of mixed among each mixed patient's --knn nearest
     neighbours, against the share of mixed patients in the cohort (enrichment)
  Distances differ in scale between OB, Jaccard and Resnik, so only the scale-free
  measures (3, 4) are comparable across runs.
Also: number of liver-side and gut-side terms reported per patient, by group (are the two
sides described with similar numbers of terms?).

Output: printed report; -o one row per patient (scores, group, term counts, cluster per run)
for plot_myo5b_scores.R.

Usage (in ~/261027_MYO5B/min8):
  python3.14 mixed_diagnostic.py --patients ~/GITHUB/pairD_phenome/261013_patientData.json \
      OB=ob_nearest Jaccard=jaccard Resnik=resnik -o myo5b_mixed.csv
"""
import argparse
import csv
import glob
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

TEMPLATE = {
    "_comment": "liver/gut: signature features (terms match their descendants). liver_branches/"
                "gut_branches: used only to count liver-side and gut-side terms per patient; a term "
                "under a liver branch counts as liver even if it is also under a gut branch.",
    "liver": ["HP:0001396", "HP:0000989", "HP:0000952", "HP:0002904", "HP:0012202"],
    "gut": ["HP:0002014", "HP:0033995", "HP:0033994", "HP:0011473"],
    "liver_branches": ["HP:0001392", "HP:0002904", "HP:0012202", "HP:0002910", "HP:0030948", "HP:0000989"],
    "gut_branches": ["HP:0025031", "HP:0033994"],
}
GROUPS = ["mixed", "liver only", "gut only", "other"]


def code(x):
    if isinstance(x, (list, tuple)):
        x = x[0]
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


def load_run(folder):
    folder = Path(folder)
    cj = folder / "clusters.json"
    if not cj.exists():
        sys.exit(f"{cj} not found.")
    d = json.load(open(cj, encoding="utf-8"))
    assign = {}
    for c in d["clustering_results"]["optimal"]["clusters"]:
        cid = c.get("cluster_id")
        for p in c.get("patients") or []:
            pid = str(p.get("id") if isinstance(p, dict) else p)
            assign[pid] = "noise" if cid == -1 else f"C{cid}"
    cdms = sorted(glob.glob(str(folder / "*" / "cdm.csv")))
    if len(cdms) != 1:
        sys.exit(f"{folder}: expected one */cdm.csv, found {len(cdms)}: {cdms}")
    dist = {}
    with open(cdms[0], newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            a, b, v = str(r["ID1"]), str(r["ID2"]), float(r["d"])
            dist[(a, b)] = dist[(b, a)] = v
    return assign, dist


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="LABEL=run folder (with clusters.json and */cdm.csv)")
    ap.add_argument("--patients", help="patient database JSON")
    ap.add_argument("--hpo-terms", default="hpo_terms.json")
    ap.add_argument("--features", default=None, help="features JSON (default: built-in, see --write-template)")
    ap.add_argument("--write-template", metavar="PATH")
    ap.add_argument("--min-liver", type=int, default=2)
    ap.add_argument("--min-gut", type=int, default=2)
    ap.add_argument("--knn", type=int, default=5)
    ap.add_argument("-o", "--output", default=None, help="per-patient CSV (for plot_myo5b_scores.R)")
    args = ap.parse_args()

    if args.write_template:
        json.dump(TEMPLATE, open(args.write_template, "w", encoding="utf-8"), indent=2)
        print(f"Wrote {args.write_template}")
        return
    if not args.runs or not args.patients:
        sys.exit("Give --patients and at least one LABEL=run folder (or --write-template).")

    H = HPO(args.hpo_terms)
    F = json.load(open(args.features, encoding="utf-8")) if args.features else TEMPLATE
    liver, gut = F["liver"], F["gut"]
    lb, gb = set(F.get("liver_branches", [])), set(F.get("gut_branches", []))
    print(f"Liver features (mixed: >= {args.min_liver}): " + ", ".join(H.name(f) for f in liver))
    print(f"Gut features   (mixed: >= {args.min_gut}): " + ", ".join(H.name(f) for f in gut))

    db = json.load(open(args.patients, encoding="utf-8"))
    db = db["patients"] if isinstance(db, dict) and "patients" in db else db
    sym = {str(p["id"]): sorted({code(s) for s in p.get("symptoms") or []}) for p in db}

    runs = {}
    for item in args.runs:
        if "=" not in item:
            sys.exit(f"Give runs as LABEL=folder, not {item!r}")
        label, folder = item.split("=", 1)
        runs[label] = load_run(folder)

    # the cohort: patients present in every run
    cohort = sorted(set.intersection(*(set(a) for a, _ in runs.values())))
    missing = [p for p in cohort if p not in sym]
    if missing:
        sys.exit(f"{len(missing)} clustered patients not in the database (e.g. {missing[0]}).")

    info = {}
    for pid in cohort:
        ups = [H.up(t) for t in sym[pid]]
        nl = sum(any(f in u for u in ups) for f in liver)
        ng = sum(any(f in u for u in ups) for f in gut)
        n_liv = sum(bool(u & lb) for u in ups)
        n_gut = sum(bool(u & gb) and not (u & lb) for u in ups)
        if nl >= args.min_liver and ng >= args.min_gut:
            g = "mixed"
        elif nl >= args.min_liver:
            g = "liver only"
        elif ng >= args.min_gut:
            g = "gut only"
        else:
            g = "other"
        info[pid] = dict(liver_score=nl / len(liver), gut_score=ng / len(gut), group=g,
                         n_liver_terms=n_liv, n_gut_terms=n_gut, n_terms=len(sym[pid]))
    sizes = Counter(info[p]["group"] for p in cohort)
    print(f"\n{len(cohort)} patients: " + ", ".join(f"{g} {sizes.get(g, 0)}" for g in GROUPS))

    print("\nTerms reported per patient (median [IQR]): liver-side | gut-side | all")
    for g in GROUPS:
        m = [info[p] for p in cohort if info[p]["group"] == g]
        if len(m) < 2:
            continue
        def q(key):
            v = sorted(x[key] for x in m)
            return f"{st.median(v):g} [{v[len(v) // 4]:g}-{v[(3 * len(v)) // 4]:g}]"
        print(f"  {g:<11} {q('n_liver_terms'):>14} | {q('n_gut_terms'):>12} | {q('n_terms'):>12}")

    mixed = [p for p in cohort if info[p]["group"] == "mixed"]
    liv = [p for p in cohort if info[p]["group"] == "liver only"]
    for label, (assign, dist) in runs.items():
        print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
        clusters = sorted({assign[p] for p in cohort}, key=lambda x: (x == "noise", x))
        print(f"  {'group':<12}" + "".join(f"{c:>7}" for c in clusters))
        for g in GROUPS:
            row = Counter(assign[p] for p in cohort if info[p]["group"] == g)
            if row:
                print(f"  {g:<12}" + "".join(f"{row.get(c, 0):>7}" for c in clusters))

        def d(a, b):
            return dist.get((a, b), float("nan"))
        present = [g for g in GROUPS if sizes.get(g, 0) >= 2]
        print(f"\n  mean distance {'':<4}" + "".join(f"{g:>12}" for g in present))
        for g1 in present:
            m1 = [p for p in cohort if info[p]["group"] == g1]
            vals = []
            for g2 in present:
                m2 = [p for p in cohort if info[p]["group"] == g2]
                vals.append(mean([d(a, b) for a in m1 for b in m2 if a != b]))
            print(f"  {g1:<18}" + "".join(f"{v:>12.3f}" for v in vals))

        if len(mixed) >= 2 and liv:
            sil = []
            for p in mixed:
                a = mean([d(p, q) for q in mixed if q != p])
                b = mean([d(p, q) for q in liv])
                sil.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
            pos = sum(s > 0 for s in sil)
            print(f"\n  mixed vs liver only: silhouette > 0 (closer to other mixed patients) in {pos}/{len(sil)} "
                  f"mixed patients; median silhouette {st.median(sil):+.3f}")
            k = min(args.knn, len(cohort) - 1)
            shares = []
            for p in mixed:
                nn = sorted((q for q in cohort if q != p), key=lambda q: d(p, q))[:k]
                shares.append(sum(info[q]["group"] == "mixed" for q in nn) / k)
            base = (len(mixed) - 1) / (len(cohort) - 1)
            print(f"  {k} nearest neighbours of mixed patients: {mean(shares):.0%} mixed "
                  f"(expected by chance {base:.0%}; enrichment x{mean(shares) / base:.1f})")
            nn_groups = Counter()
            for p in mixed:
                nn = sorted((q for q in cohort if q != p), key=lambda q: d(p, q))[:k]
                nn_groups.update(info[q]["group"] for q in nn)
            tot = sum(nn_groups.values())
            print("  their neighbours by group: " + ", ".join(f"{g} {nn_groups.get(g, 0) / tot:.0%}" for g in GROUPS))
        for p in cohort:
            info[p][f"cluster_{label}"] = assign[p]

    if args.output:
        art = {}
        for p in db:
            a = [x for x in p.get("articles") or [] if isinstance(x, dict)]
            art[str(p["id"])] = (str(a[0].get("doi") or "").replace("https://doi.org/", "") if a else "")
        keys = ["liver_score", "gut_score", "group", "n_liver_terms", "n_gut_terms", "n_terms"] + \
               [f"cluster_{r}" for r in runs]
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "article"] + keys)
            for p in cohort:
                w.writerow([p, art.get(p, "")] + [info[p][k] for k in keys])
        print(f"\nPer-patient table: {args.output}")


if __name__ == "__main__":
    main()
