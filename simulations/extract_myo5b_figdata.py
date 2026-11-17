#!/usr/local/bin/python3.14
"""
Extracts the data for the MYO5B figures from one pairD_phenome.py run (one distance) into
tidy CSV files, read by fig_myo5b_umap.R, fig_myo5b_phenotype.R and fig_myo5b_genotype.R.

Inputs
  --run DIR       a run folder: DIR/clusters.json and DIR/<date>_MYO5B_*/ (cdm.csv,
                  stability.csv, few_feature_flags.csv), e.g. 5_min8_mindist0_flag/ob_nearest
  --patients      the patient database (261013_patientData.json)
  --hpo-terms     hpo_terms.json (custom HPO)
  --expected      expected_myo5b.json (signature terms; used to name the clusters and to
                  mark the patients' signatures)
  --solution      optimal (default) or second
  -o PREFIX       output prefix (default: <run folder name>, e.g. ob_nearest)

Outputs (PREFIX_*.csv)
  _patients.csv    one row per patient: cluster, 2-D UMAP coordinates, signatures present,
                   article, publication year, genotyped or not, genotype class,
                   vital status and age at death / last news
  _clusters.csv    one row per cluster: name, size, resampling stability, few-feature flag
  _prevalence.csv  cluster x HPO term prevalence (as reported, as pairD counts them), for the
                   most discriminating terms, with an organ group and subgroup
  _absent.csv      cluster x HPO term: % of patients in whom the symptom is reported absent
                   (listed in absentsymptoms and not in symptoms; intermittent symptoms, coded
                   as both, are left out), for terms absent in >= --min-absent %% of a cluster
  _variants.csv    one row per allele with a protein position: patient, cluster, change,
                   type, position (for the protein map)

2-D UMAP: pairD clusters in a 10-D UMAP embedding. For the figure, the same distance matrix
(cdm.csv) is embedded in 2-D with pairD's settings (precomputed metric, n_neighbors 30,
min_dist 0, random_state 25). This is a view of the data; the clusters come from the 10-D
embedding. If umap-learn is not importable, the first two of the 10 dimensions are used
instead (a warning is printed).

Usage (in ~/pairD_eval/myo5b):
  for d in ob_nearest jaccard resnik; do
    python3.14 ../scripts/extract_myo5b_figdata.py --run 5_min8_mindist0_flag/$d \\
        --patients ~/GITHUB/pairD_phenome/261013_patientData.json \\
        --hpo-terms ../inputs/hpo_terms.json --expected expected_myo5b.json \\
        -o figdata/$d
  done
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = "HP:0000118"
FIBROSIS = "HP:0001395"   # hepatic fibrosis (portal, periportal, bridging ... below it)
# A patient has a signature when ALL its items are present; an item that is a list is met by
# ANY of its terms. Each term is counted with its descendants.
SIGNATURES = {
    # diarrhoea, microvillus inclusions, dependency on parenteral nutrition
    "intestinal": ["HP:0002014", "HP:0033995", "HP:0033994"],
    # cholestasis, pruritus, hyperbilirubinaemia, elevated AST or ALT
    "cholestatic": ["HP:0001396", "HP:0000989", "HP:0002904", ["HP:0031956", "HP:0031964"]],
}
# organ groups and subgroups for the phenotype figure: (group, subgroup, root terms); a term
# goes to the first entry with one of the roots among its ancestors (edit freely). Terms that
# match none are grouped by their top-level HPO system.
GROUPS = [
    ("Intestine", "Diarrhoea", ["HP:0002014"]),
    ("Intestine", "Mucosal histology", ["HP:0011472", "HP:6000337"]),
    ("Intestine", "Nutrition and absorption", ["HP:0033994", "HP:0002024", "HP:0011968"]),
    ("Liver", "Cholestasis", ["HP:0001396", "HP:0002904", "HP:0012202", "HP:0000989", "HP:0003155",
                              "HP:0030948"]),
    ("Liver", "Hepatocellular injury", ["HP:0002910"]),
    ("Liver", "Liver histology", ["HP:0001395", "HP:0001394", "HP:0200084", "HP:0033196", "HP:0001408"]),
    ("Liver", "Organomegaly", ["HP:0002240", "HP:0001744"]),
    ("Liver", "Other liver", ["HP:0001392"]),
    ("Intestine", "Other intestinal", ["HP:0011024", "HP:0025032"]),
    ("Growth", "Growth", ["HP:0001507"]),
    ("Prenatal and birth", "Prenatal and birth", ["HP:0001197"]),
    ("Kidney", "Kidney", ["HP:0000077", "HP:0012211"]),
    ("Metabolism", "Fluid, protein and acid-base", ["HP:0001939"]),
    ("Blood", "Blood and coagulation", ["HP:0001871"]),
]
NULL_TYPES = {"nonsense", "frameshift", "splicing"}
POS_RE = re.compile(r"p\.\[?\(?(?:[A-Z][a-z]{2}|[A-Z])(\d+)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--patients", required=True)
    ap.add_argument("--hpo-terms", required=True)
    ap.add_argument("--expected", required=True)
    ap.add_argument("--solution", default="optimal", choices=["optimal", "second"])
    ap.add_argument("-o", "--output")
    ap.add_argument("--max-terms", type=int, default=36)
    ap.add_argument("--min-absent", type=float, default=10.0,
                    help="absent-symptom figure: minimum %% of a cluster reporting the symptom absent")
    ap.add_argument("--min-range", type=float, default=20.0,
                    help="minimum prevalence difference between clusters (%%) for a term to be shown")
    args = ap.parse_args()

    run = Path(args.run)
    out = Path(args.output or run.name)
    out.parent.mkdir(parents=True, exist_ok=True)
    res_dirs = sorted(run.glob("*_MYO5B_*"))
    if not (run / "clusters.json").exists() or not res_dirs:
        sys.exit(f"{run}: clusters.json or the <date>_MYO5B_* result folder is missing (run not finished?).")
    res = res_dirs[-1]

    # ---- HPO -------------------------------------------------------------------------
    terms = json.load(open(args.hpo_terms, encoding="utf-8"))["terms"]
    norm = lambda c: "HP:" + str(c).replace("HP:", "").zfill(7)
    name = lambda c: terms.get(c, {}).get("name", c)
    anc_cache = {}

    def ancestors(c):            # self included
        if c not in anc_cache:
            seen, stack = {c}, [c]
            while stack:
                for p in terms.get(stack.pop(), {}).get("parents", []):
                    if p not in seen:
                        seen.add(p); stack.append(p)
            anc_cache[c] = seen
        return anc_cache[c]

    def group(c):
        a = ancestors(c)
        for g, sub, roots in GROUPS:
            if a & set(roots):
                return g, sub
        for top in terms.get(ROOT, {}).get("children", []):
            if top in a:
                g = name(top).replace("Abnormality of the ", "").replace("Abnormality of ", "")
                return "Other", g[0].upper() + g[1:]
        return "Other", "Other"

    # ---- clusters ----------------------------------------------------------------------
    sol = json.load(open(run / "clusters.json"))["clustering_results"][args.solution]
    col = sol["clustering_column"]
    db = {p["id"]: p for p in json.load(open(args.patients, encoding="utf-8"))}
    expected = json.load(open(args.expected))
    sig_intest = set(expected["MVID"]["present"])
    sig_chol = set(expected["PFIC"]["present"])

    pts = []
    for c in sol["clusters"]:
        for p in c.get("patients") or []:
            pid = p["id"] if isinstance(p, dict) else p
            codes = {norm(s) for s in (p.get("symptoms") if isinstance(p, dict) else None)
                     or [s[0] for s in db[pid]["symptoms"]]}
            pts.append({"id": pid, "cluster_id": c["cluster_id"], "codes": codes})
    has = lambda codes, roots: any(ancestors(s) & roots for s in codes)
    for p in pts:
        for sig, req in SIGNATURES.items():
            p[sig] = all(has(p["codes"], set(t) if isinstance(t, list) else {t}) for t in req)
        p["fibrosis"] = has(p["codes"], {FIBROSIS})

    # cluster names: the expected_myo5b.json terms, ANY present (>= 50 % of patients); kept
    # separate from the stricter figure signatures above, so the names do not change with them
    for p in pts:
        p["_intest_any"] = has(p["codes"], sig_intest)
        p["_chol_any"] = has(p["codes"], sig_chol)
    cids = sorted({p["cluster_id"] for p in pts if p["cluster_id"] != -1})
    names = {-1: "Noise"}
    for cid in cids:
        mem = [p for p in pts if p["cluster_id"] == cid]
        i = np.mean([p["_intest_any"] for p in mem]) >= 0.5
        h = np.mean([p["_chol_any"] for p in mem]) >= 0.5
        names[cid] = "MVID+PFIC" if i and h else "MVID" if i else "PFIC" if h else f"Cluster {cid}"
    dup = Counter(names[c] for c in cids)
    seen = Counter()
    for cid in cids:                       # e.g. two MVID clusters -> MVID a, MVID b
        if dup[names[cid]] > 1:
            seen[names[cid]] += 1
            names[cid] = f"{names[cid]} {'abcdef'[seen[names[cid]] - 1]}"

    # ---- 2-D embedding -----------------------------------------------------------------
    cdm = pd.read_csv(res / "cdm.csv")
    order = [cdm["ID1"].iloc[0]] + list(cdm.loc[cdm["ID1"] == cdm["ID1"].iloc[0], "ID2"])
    from scipy.spatial.distance import squareform
    X = squareform(cdm["d"].to_numpy())
    try:
        from umap import UMAP
        emb = UMAP(n_components=2, metric="precomputed", n_neighbors=min(30, len(order) - 1),
                   min_dist=0.0, random_state=25).fit_transform(X)
        emb_src = "2-D UMAP of the distance matrix (n_neighbors 30, min_dist 0, seed 25)"
    except ImportError:
        u = pd.read_csv(res / "pairD_UMAP.csv").set_index("ids")
        emb = u.loc[order, ["UMAP_1", "UMAP_2"]].to_numpy()
        emb_src = "first two dimensions of pairD's 10-D UMAP (umap-learn not importable)"
        print("WARNING: umap-learn not importable; using UMAP_1 / UMAP_2 of the 10-D embedding.")
    xy = {pid: emb[i] for i, pid in enumerate(order)}

    # ---- per-patient table -------------------------------------------------------------
    def gclass(p):
        if any("Not tested" in g for g in p.get("gene") or []):
            return "Not genotyped"
        t = [str(x) for x in (p.get("protvartypes") or [])]
        if len(t) < 2 or "unknown" in t:
            return "Incomplete"
        n_null = sum(x in NULL_TYPES for x in t)
        n_mis = sum(x == "missense" for x in t)
        if n_null == 2:
            return "Biallelic null"
        if n_null == 1 and n_mis == 1:
            return "Null + missense"
        if n_mis == 2:
            return "Biallelic missense"
        return "Other"

    rows, variants = [], []
    for p in pts:
        d = db[p["id"]]
        art = (d.get("articles") or [{}])[0]
        rows.append({
            "id": p["id"], "cluster_id": p["cluster_id"], "cluster": names[p["cluster_id"]],
            "umap_x": round(float(xy[p["id"]][0]), 4), "umap_y": round(float(xy[p["id"]][1]), 4),
            "n_terms": len(p["codes"]),
            "intestinal": p["intestinal"], "cholestatic": p["cholestatic"], "fibrosis": p["fibrosis"],
            "doi": art.get("doi", ""), "year": art.get("publication_year"),
            "genotyped": gclass(d) != "Not genotyped", "genotype_class": gclass(d),
            "zygosity": d.get("zygosity"),
            "vital": d.get("alivedead"), "age_years": d.get("alivedeadage") if d.get("alivedeadage") is not None
            else d.get("lastnewsageyear"),
            "onset_months": d.get("firstsymptomagemonth"),
        })
        types = d.get("protvartypes") or []
        for k, key in enumerate(("protvariant1", "protvariant2")):
            v = d.get(key)
            change = v.get("clean_p") if isinstance(v, dict) else v
            m = POS_RE.match(str(change or ""))
            if m:
                variants.append({"id": p["id"], "cluster": names[p["cluster_id"]], "allele": k + 1,
                                 "change": change, "type": types[k] if k < len(types) else "unknown",
                                 "position": int(m.group(1))})
    pd.DataFrame(rows).to_csv(f"{out}_patients.csv", index=False)
    # one row per distinct variant per patient (a homozygous variant counted once)
    vdf = pd.DataFrame(variants)
    if len(vdf):
        vdf = vdf.drop_duplicates(["id", "change"])
    vdf.to_csv(f"{out}_variants.csv", index=False)

    # ---- clusters ------------------------------------------------------------------------
    stab = pd.read_csv(res / "stability.csv") if (res / "stability.csv").exists() else None
    crow = []
    for c in sol["clusters"]:
        cid = c["cluster_id"]
        crow.append({"cluster_id": cid, "cluster": names[cid], "n": c["patient_count"],
                     "stability": c.get("stability_jaccard_mean"), "dissolved": c.get("stability_dissolved"),
                     "flag_few_features": c.get("flag_few_features"),
                     "n_discriminating": c.get("n_discriminating_features"),
                     "closest": names.get(c.get("closest_cluster"), "")})
    pd.DataFrame(crow).to_csv(f"{out}_clusters.csv", index=False)

    # ---- prevalence ----------------------------------------------------------------------
    real = [cid for cid in cids]
    mem = {cid: [p for p in pts if p["cluster_id"] == cid] for cid in real}
    prev = {}
    for cid in real:
        cnt = Counter(s for p in mem[cid] for s in p["codes"])
        prev[cid] = {s: 100 * v / len(mem[cid]) for s, v in cnt.items()}
    allt = set().union(*prev.values())
    rng = {s: max(prev[c].get(s, 0) for c in real) - min(prev[c].get(s, 0) for c in real) for s in allt}
    shown = [s for s in sorted(allt, key=lambda s: -rng[s]) if rng[s] >= args.min_range][: args.max_terms]
    prow = []
    for s in shown:
        for cid in real:
            prow.append({"group": group(s)[0], "subgroup": group(s)[1], "hpo": s, "term": name(s),
                         "cluster": names[cid],
                         "prevalence": round(prev[cid].get(s, 0), 1), "range": round(rng[s], 1)})
    pd.DataFrame(prow).to_csv(f"{out}_prevalence.csv", index=False)

    # ---- explicitly absent symptoms (cross-validation; not used for clustering) ---------------
    # A symptom is truly absent in a patient when it is listed in "absentsymptoms" and not in
    # "symptoms" (intermittent symptoms are deliberately coded as both, and are left out).
    absent = {}
    for p in pts:
        d = db[p["id"]]
        listed = {norm(s[0] if isinstance(s, (list, tuple)) else s) for s in d.get("absentsymptoms") or []}
        absent[p["id"]] = listed - p["codes"]
    aprev = {}
    for cid in real:
        cnt = Counter(s for p in mem[cid] for s in absent[p["id"]])
        aprev[cid] = {s: 100 * v / len(mem[cid]) for s, v in cnt.items()}
    alla = set().union(*aprev.values())
    amax = {s: max(aprev[c].get(s, 0) for c in real) for s in alla}
    ashown = [s for s in sorted(alla, key=lambda s: -amax[s]) if amax[s] >= args.min_absent][: args.max_terms]
    arow = []
    for s in ashown:
        for cid in real:
            arow.append({"group": group(s)[0], "subgroup": group(s)[1], "hpo": s, "term": name(s),
                         "cluster": names[cid], "prevalence": round(aprev[cid].get(s, 0), 1),
                         "range": round(amax[s], 1)})
    pd.DataFrame(arow).to_csv(f"{out}_absent.csv", index=False)
    n_abs = sum(bool(v) for v in absent.values())

    # ---- console -------------------------------------------------------------------------
    df = pd.DataFrame(rows)
    print(f"{run}: solution '{args.solution}' ({col}), {len(df)} patients; embedding: {emb_src}")
    print(pd.DataFrame(crow).to_string(index=False))
    print("\nGenotype class by cluster:")
    print(pd.crosstab(df.cluster, df.genotype_class))
    print("\nVital status by cluster:")
    print(pd.crosstab(df.cluster, df.vital.fillna("unknown")))
    print(f"\nWrote {out}_patients.csv, _clusters.csv, _prevalence.csv, _absent.csv, _variants.csv "
          f"({len(shown)} discriminating terms, {len(ashown)} absent terms from {n_abs} patients "
          f"with truly absent symptoms, {len(vdf)} variants with a position)")


if __name__ == "__main__":
    main()
