#!/usr/local/bin/python3.14
"""
TEST 1 (repeat): builds the symptom tables of several independent compositions (C1, C2,
C3, ...) of the three artificial diseases D1S0 / D2S2 / D3S3, from one pool of Orphanet
symptom lists.

  * The compositions share NO symptom: every symptom is used by one composition at most.
  * They are filled in rotation, slot by slot (C1, C2, C3, then C2, C3, C1, ...), so that
    no composition gets the leftovers: all draw from the same quality of candidates and
    have the same organ-system structure.

Each composition table contains
  core        --n-core symptoms shared by all patients; every organ system has one
              "anchor" (most frequent available, >= 0.80 when possible)
  dispersed   signature sets A, B, C, ranks 1..k_max; at each rank the three symptoms
              come from the same organ system, systems taken in turn (as in the first run)
  concentrated signature sets A, B, C, ranks 1..k_conc: ALL symptoms of set A come from
              one home organ system, of B from a second, of C from a third
              (--homes, default nerves, eye, head/neck)

Rules (as build_test1_table.py): every symptom belongs to exactly one of the chosen
organ systems; sex-specific and very general terms are excluded; within a composition no
symptom is an ancestor or descendant of another core or same-layout signature symptom.
A concentrated signature symptom may also be a dispersed signature symptom of the same
composition (the two layouts are never simulated together).

Output: <prefix>_C1.csv, <prefix>_C2.csv, ... (columns as test1_table.csv plus 'layout'),
read by generate_test1_patients.py --layout dispersed|concentrated.

Usage:
  python3.14 build_test1_tables.py -j hpo_terms.json --seed 1 --n-comp 3 -o test1_table \\
      symptom_list_*.csv
"""
import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import build_test1_table as b1   # pooling, ancestors, organ systems, sex-specific branches

SETS = "ABC"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("symptom_csvs", nargs="+")
    ap.add_argument("-j", "--hpo-terms", default="hpo_terms.json")
    ap.add_argument("-o", "--prefix", default="test1_table")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-comp", type=int, default=3, help="Number of compositions.")
    ap.add_argument("--n-core", type=int, default=20)
    ap.add_argument("--k-max", type=int, default=9, help="Dispersed signature ranks.")
    ap.add_argument("--k-conc", type=int, default=9, help="Concentrated signature ranks.")
    ap.add_argument("--systems", default=b1.DEFAULT_SYSTEMS)
    ap.add_argument("--homes", default="nerves,eye,head/neck",
                    help="Home organ systems of concentrated sets A, B, C.")
    ap.add_argument("--exclude", nargs="*", default=[], metavar="HP:XXXXXXX")
    ap.add_argument("--max-descendants", type=int, default=40)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    terms, alt = b1.load_hpo(args.hpo_terms)
    pool = b1.pool_symptoms(args.symptom_csvs, terms, alt)
    systems = [s.strip() for s in args.systems.split(",")]
    homes = [s.strip() for s in args.homes.split(",")]
    sys_code = dict(b1.ORGAN_SYSTEMS)
    if any(s not in sys_code for s in systems + homes) or len(homes) != 3 or len(set(homes)) != 3:
        sys.exit("Unknown organ system, or --homes is not three different systems.")
    if not set(homes) <= set(systems):
        sys.exit("Home systems must be among --systems (so each home also has core symptoms).")

    banned = {c if c.startswith("HP:") else f"HP:{c}" for c in args.exclude} | set(b1.SEX_SPECIFIC)
    anc_memo, desc_memo = {}, {}
    anc = {c: b1.ancestors(c, terms, anc_memo) for c in pool}
    candidates = defaultdict(list)
    for c in sorted(pool):
        if c in banned or anc[c] & banned:
            continue
        if b1.n_descendants(c, terms, desc_memo) > args.max_descendants:
            continue
        in_sys = [s for s, oc in b1.ORGAN_SYSTEMS if f"HP:{oc}" in anc[c]]
        if len(in_sys) != 1 or in_sys[0] not in systems:
            continue
        pool[c]["freq"] = round(sum(pool[c]["freqs"]) / len(pool[c]["freqs"]), 2)
        pool[c]["system"] = in_sys[0]
        candidates[in_sys[0]].append(c)
    for s in systems:
        rng.shuffle(candidates[s])

    comps = [f"C{i + 1}" for i in range(args.n_comp)]
    used_by = {}                                   # symptom -> composition
    rows = {c: [] for c in comps}                  # composition -> rows
    chosen = {c: {"core": [], "dispersed": [], "concentrated": []} for c in comps}
    slot = [0]

    def related(a, b):
        return a in anc[b] or b in anc[a]

    def ok(comp, code, layout):
        owner = used_by.get(code)
        if owner is not None and owner != comp:
            return False
        ch = chosen[comp]
        others = ch["core"] + (ch[layout] if layout != "core" else ch["dispersed"] + ch["concentrated"])
        if code in ch["core"] or (layout != "core" and code in ch[layout]):
            return False
        if layout == "core" and (code in ch["dispersed"] or code in ch["concentrated"]):
            return False
        return all(x == code or not related(code, x) for x in others)

    def pick(comp, system, layout, cond=lambda c: True, best_freq=False):
        cands = [c for c in candidates[system] if ok(comp, c, layout) and cond(c)]
        if not cands:
            return None
        c = max(cands, key=lambda x: pool[x]["freq"]) if best_freq else cands[0]
        used_by[c] = comp
        chosen[comp][layout].append(c)
        return c

    def rotation():
        k = slot[0] % len(comps)
        slot[0] += 1
        return comps[k:] + comps[:k]

    def fail(what):
        sys.exit(f"Not enough symptoms for {what}. Lower --k-max/--k-conc/--n-core, change --homes, "
                 f"or add symptom lists.")

    # 1. anchors, one per system per composition
    for s in systems:
        for comp in rotation():
            c = pick(comp, s, "core", lambda x: pool[x]["freq"] >= 0.80)
            if c is None:
                c = pick(comp, s, "core", best_freq=True)
                if c is None:
                    fail(f"the {s} anchor of {comp}")
                print(f"  note: {comp} {s} anchor below 0.80 ({pool[c]['label']}, {pool[c]['freq']:.2f})",
                      file=sys.stderr)
            rows[comp].append({"hpo": c, "role": "core", "set": "", "rank": "", "anchor": "yes", "layout": ""})

    # 2. concentrated signatures (scarcest: home systems), then 3. dispersed signatures
    for r in range(1, args.k_conc + 1):
        for st, home in zip(SETS, homes):
            for comp in rotation():
                c = pick(comp, home, "concentrated")
                if c is None:
                    fail(f"concentrated rank {r} of set {st} ({home}) in {comp}")
                rows[comp].append({"hpo": c, "role": "signature", "set": st, "rank": r, "anchor": "",
                                   "layout": "concentrated"})
    for r in range(1, args.k_max + 1):
        s = systems[(r - 1) % len(systems)]
        for st in SETS:
            for comp in rotation():
                c = pick(comp, s, "dispersed")
                if c is None:
                    fail(f"dispersed rank {r} of set {st} ({s}) in {comp}")
                rows[comp].append({"hpo": c, "role": "signature", "set": st, "rank": r, "anchor": "",
                                   "layout": "dispersed"})

    # 4. rest of the core, systems in turn
    for i in range(args.n_core - len(systems)):
        s = systems[i % len(systems)]
        for comp in rotation():
            c = pick(comp, s, "core")
            if c is None:
                fail(f"core symptom {len(systems) + i + 1} ({s}) in {comp}")
            rows[comp].append({"hpo": c, "role": "core", "set": "", "rank": "", "anchor": "", "layout": ""})

    # final check: no symptom in two compositions
    seen = {}
    for comp in comps:
        for r in rows[comp]:
            if seen.get(r["hpo"], comp) != comp:
                sys.exit(f"BUG: {r['hpo']} in {seen[r['hpo']]} and {comp}")
            seen[r["hpo"]] = comp

    fields = ["hpo", "label", "role", "layout", "set", "rank", "organ_system", "core_freq", "anchor",
              "orphanet_freq", "sources"]
    order = {"": 0, "dispersed": 1, "concentrated": 2}
    for comp in comps:
        out = Path(f"{args.prefix}_{comp}.csv")
        rs = sorted(rows[comp], key=lambda r: (order[r["layout"]], r["set"], r["rank"] or 0,
                                               pool[r["hpo"]]["system"]))
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rs:
                e = pool[r["hpo"]]
                w.writerow({**r, "label": e["label"], "organ_system": e["system"],
                            "core_freq": e["freq"] if r["role"] == "core" else "",
                            "orphanet_freq": e["freq"], "sources": "/".join(sorted(set(e["sources"])))})

        core = [r for r in rs if r["role"] == "core"]
        print(f"\n=== {comp}: {len(core)} core symptoms, {sum(pool[r['hpo']]['freq'] for r in core):.1f} expected "
              f"per patient; anchors: " + ", ".join(f"{pool[r['hpo']]['freq']:.2f}" for r in core if r["anchor"]))
        for layout, kk in (("dispersed", args.k_max), ("concentrated", args.k_conc)):
            print(f"  {layout} (rank: A | B | C)")
            for rk in range(1, kk + 1):
                trio = [next(x for x in rs if x["layout"] == layout and x["rank"] == rk and x["set"] == st)
                        for st in SETS]
                print(f"   {rk:2}  " + " | ".join(f"{pool[t['hpo']]['label'][:28]} [{pool[t['hpo']]['system']}]"
                                                for t in trio))
        print(f"  -> {out}")

    print(f"\nPool: {len(pool)} symptoms; usable per system: " +
          ", ".join(f"{s} {len(candidates[s])}" for s in systems))
    print(f"Used: {len(used_by)} symptoms over {len(comps)} compositions (none shared).")


if __name__ == "__main__":
    main()
