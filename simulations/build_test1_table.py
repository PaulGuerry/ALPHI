#!/usr/local/bin/python3.14
"""
TEST 1: proposes the symptom table of three artificial diseases built from Orphanet
symptom lists.

  D1S0  no subgroups     core + signature symptoms at one flat frequency (null case)
  D2S2  two subgroups    core + signature sets A, B
  D3S3  three subgroups  core + signature sets A, B, C

The table fixes WHICH symptoms are used and their organ systems. The subgroup
frequencies of the signature symptoms (high in their own subgroup, low in the others)
and the signature set size k are chosen when patients are generated
(generate_test1_patients.py), so the same table serves every grid point.

Design rules
  * Organ systems: --systems (default 8). Every selected symptom belongs to exactly one
    of them, so organ-system membership is unambiguous.
  * Core (--n-core, default 20): every system has one "anchor" core symptom at >= 0.80
    (Orphanet very frequent), so signature symptoms are rarely alone in their organ
    system; the rest of the core is drawn across the systems with their Orphanet
    frequencies.
  * Signature sets A, B, C are ranked 1..k_max. At each rank the three symptoms come from
    the SAME organ system (systems taken in turn), so the sets are matched: a signature
    of size k uses ranks 1..k of each set.
  * Sex-specific symptoms (male or female genitalia or reproductive physiology, breast)
    are excluded, as are any codes given with --exclude.
  * No selected symptom is an ancestor or descendant of another, and very general terms
    (more than --max-descendants descendants, e.g. "Neoplasm") are excluded.

Output: a CSV (one row per symptom), to review and edit by hand. Rows can be swapped as
long as the rules above are kept; check_test1_table.py-like checks are rerun by
generate_test1_patients.py when it reads the table.

Usage:
  python3.14 build_test1_table.py -j hpo_terms.json --seed 1 -o test1_table.csv \
      symptom_list_alagille.csv symptom_list_bws.csv symptom_list_marfan.csv \
      symptom_list_wilson.csv symptom_list_gcdhd.csv symptom_list_dubowitz.csv
"""
import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ORGAN_SYSTEMS = (   # same list and codes as pgmodules.ORGAN_SYSTEMS
    ('musculoskeletal', '0033127'), ('limbs', '0040064'), ('nerves', '0000707'),
    ('metabolism', '0001939'), ('genitourinary', '0000119'), ('cellular', '0025354'),
    ('immune', '0002715'), ('eye', '0000478'), ('integument', '0001574'),
    ('voice', '0001608'), ('digestive', '0025031'), ('ear', '0000598'),
    ('cardiovascular', '0001626'), ('prenatal', '0001197'), ('neoplasm', '0002664'),
    ('head/neck', '0000152'), ('endocrine', '0000818'), ('constitutional', '0025142'),
    ('growth', '0001507'), ('thoracic', '0045027'), ('respiratory', '0002086'),
    ('blood', '0001871'), ('breast', '0000769'),
)
# Symptoms that can only occur in one sex are left out (every simulated patient must be
# able to carry every symptom). Terms under these branches are excluded:
SEX_SPECIFIC = {
    "HP:0010461": "Abnormality of the male genitalia",
    "HP:0010460": "Abnormality of the female genitalia",
    "HP:0012874": "Abnormal male reproductive system physiology",
    "HP:0030012": "Abnormal female reproductive system physiology",
    "HP:0000769": "Abnormality of the breast",
    "HP:0012041": "Decreased fertility in males",
    "HP:0000868": "Decreased fertility in females",
}
DEFAULT_SYSTEMS = "musculoskeletal,nerves,head/neck,digestive,cardiovascular,eye,genitourinary,integument"
SETS = "ABC"


def load_hpo(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return d["terms"], d.get("alt_id_map", {})


def ancestors(tid, terms, memo):
    hit = memo.get(tid)
    if hit is None:
        hit = set()
        for p in terms[tid]["parents"]:
            if p in terms:
                hit |= {p} | ancestors(p, terms, memo)
        memo[tid] = hit
    return hit


def n_descendants(tid, terms, memo):
    hit = memo.get(tid)
    if hit is None:
        seen, stack = set(), list(terms[tid]["children"])
        while stack:
            c = stack.pop()
            if c in terms and c not in seen:
                seen.add(c)
                stack.extend(terms[c]["children"])
        hit = memo[tid] = len(seen)
    return hit


def pool_symptoms(paths, terms, alt):
    """{hpo: {"label", "freqs": [..], "sources": [..]}} over all lists, retired IDs resolved."""
    pool = {}
    for p in paths:
        src = Path(p).stem.replace("symptom_list_", "")
        with open(p, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                code = row["HPO Code"].strip()
                if code in alt:
                    print(f"  {src}: {code} retired -> {alt[code]}", file=sys.stderr)
                    code = alt[code]
                if code not in terms or terms[code]["obsolete"]:
                    print(f"  {src}: {code} not a current HPO term, skipped", file=sys.stderr)
                    continue
                e = pool.setdefault(code, {"label": terms[code]["name"], "freqs": [], "sources": []})
                e["freqs"].append(float(row["Frequency"]))
                e["sources"].append(src)
    return pool


def related(a, b, anc):
    return a in anc[b] or b in anc[a]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("symptom_csvs", nargs="+")
    ap.add_argument("-j", "--hpo-terms", default="hpo_terms.json")
    ap.add_argument("-o", "--output", default="test1_table.csv")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-core", type=int, default=20)
    ap.add_argument("--k-max", type=int, default=12, help="Largest signature set size to prepare (ranks 1..k_max).")
    ap.add_argument("--systems", default=DEFAULT_SYSTEMS, help="Comma-separated organ systems to use.")
    ap.add_argument("--exclude", nargs="*", default=[], metavar="HP:XXXXXXX",
                    help="HPO codes (and everything under them) to leave out.")
    ap.add_argument("--keep-sex-specific", action="store_true",
                    help="Do not exclude sex-specific symptoms.")
    ap.add_argument("--max-descendants", type=int, default=40,
                    help="Exclude terms more general than this (number of descendant terms).")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    terms, alt = load_hpo(args.hpo_terms)
    pool = pool_symptoms(args.symptom_csvs, terms, alt)
    systems = [s.strip() for s in args.systems.split(",")]
    sys_code = dict(ORGAN_SYSTEMS)
    unknown = [s for s in systems if s not in sys_code]
    if unknown:
        sys.exit(f"Unknown organ system(s): {unknown}. Choose from: {', '.join(sys_code)}")

    anc_memo, desc_memo = {}, {}
    anc = {c: ancestors(c, terms, anc_memo) for c in pool}
    candidates = defaultdict(list)          # system -> codes in that system only
    banned = {c if c.startswith("HP:") else f"HP:{c}" for c in args.exclude}
    if not args.keep_sex_specific:
        banned |= set(SEX_SPECIFIC)
    excluded = {"sex-specific/excluded": 0, "general": 0, "several systems": 0, "other systems": 0}
    for c in sorted(pool):
        if c in banned or anc[c] & banned:
            excluded["sex-specific/excluded"] += 1
            continue
        if n_descendants(c, terms, desc_memo) > args.max_descendants:
            excluded["general"] += 1
            continue
        in_sys = [s for s, oc in ORGAN_SYSTEMS if f"HP:{oc}" in anc[c] or c == f"HP:{oc}"]
        if len(in_sys) != 1:
            excluded["several systems" if in_sys else "other systems"] += 1
            continue
        if in_sys[0] not in systems:
            excluded["other systems"] += 1
            continue
        pool[c]["freq"] = round(sum(pool[c]["freqs"]) / len(pool[c]["freqs"]), 2)
        pool[c]["system"] = in_sys[0]
        candidates[in_sys[0]].append(c)
    for s in systems:
        rng.shuffle(candidates[s])

    chosen = []

    def free(c):
        return all(not related(c, x, anc) for x in chosen) and c not in chosen

    def take(system, cond=lambda c: True):
        for c in candidates[system]:
            if free(c) and cond(c):
                chosen.append(c)
                return c
        return None

    rows = []
    # 1. one very frequent anchor per system
    for s in systems:
        c = take(s, lambda c: pool[c]["freq"] >= 0.80)
        if c is None:   # no very frequent symptom: use the most frequent one available
            best = max((x for x in candidates[s] if free(x)), key=lambda x: pool[x]["freq"], default=None)
            if best is None:
                sys.exit(f"No anchor symptom available in organ system '{s}'.")
            c = take(s, lambda x: x == best)
            print(f"  note: no very frequent symptom in {s}; anchor is the most frequent available "
                  f"({pool[c]['label']}, {pool[c]['freq']:.2f})", file=sys.stderr)
        rows.append({"hpo": c, "role": "core", "set": "", "rank": "", "anchor": "yes"})

    # 2. signature sets: rank r uses system r (in turn), one symptom per set
    for r in range(1, args.k_max + 1):
        s = systems[(r - 1) % len(systems)]
        for st in SETS:
            c = take(s)
            if c is None:
                sys.exit(f"Not enough symptoms in '{s}' for rank {r} of set {st}; "
                         f"lower --k-max or add systems/lists.")
            rows.append({"hpo": c, "role": "signature", "set": st, "rank": r, "anchor": ""})

    # 3. rest of the core, systems in turn
    n_more = args.n_core - len(systems)
    i = 0
    while n_more > 0:
        s = systems[i % len(systems)]
        c = take(s)
        i += 1
        if c is not None:
            rows.append({"hpo": c, "role": "core", "set": "", "rank": "", "anchor": ""})
            n_more -= 1
        if i > 50 * len(systems):
            sys.exit("Ran out of symptoms for the core; lower --n-core.")

    fields = ["hpo", "label", "role", "set", "rank", "organ_system", "core_freq", "anchor",
              "orphanet_freq", "sources"]
    order = {"core": 0, "signature": 1}
    rows.sort(key=lambda r: (order[r["role"]], r["set"], r["rank"] or 0, pool[r["hpo"]]["system"]))
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            e = pool[r["hpo"]]
            w.writerow({**r, "label": e["label"], "organ_system": e["system"],
                        "core_freq": e["freq"] if r["role"] == "core" else "",
                        "orphanet_freq": e["freq"], "sources": "/".join(sorted(set(e["sources"])))})

    # summary
    print(f"Pooled {len(pool)} symptoms from {len(args.symptom_csvs)} lists; usable: "
          f"{sum(len(v) for v in candidates.values())} "
          f"(excluded: {excluded['sex-specific/excluded']} sex-specific or excluded, {excluded['general']} too general, {excluded['several systems']} in several "
          f"systems, {excluded['other systems']} outside the chosen systems)")
    for s in systems:
        print(f"  {s:<16} {len(candidates[s]):3d} usable")
    core = [r for r in rows if r["role"] == "core"]
    exp = sum(pool[r["hpo"]]["freq"] for r in core)
    print(f"\nCore: {len(core)} symptoms, {exp:.1f} expected per patient")
    for r in core:
        e = pool[r["hpo"]]
        print(f"  {e['freq']:.2f}  {e['system']:<16} {r['hpo']} {e['label']}{'  (anchor)' if r['anchor'] else ''}")
    print(f"\nSignature sets (rank: A | B | C), k_max = {args.k_max}")
    for rk in range(1, args.k_max + 1):
        trio = [next(x for x in rows if x["rank"] == rk and x["set"] == st) for st in SETS]
        print(f"  {rk}  {pool[trio[0]['hpo']]['system']:<16} " +
              " | ".join(pool[t['hpo']]['label'] for t in trio))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
