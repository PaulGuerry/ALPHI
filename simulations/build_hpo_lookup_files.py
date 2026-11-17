#!/usr/local/bin/python3.14
"""
Build the two HPO lookup files used by pairD_phenome.py / pgmodules.py from a single
source, hpo_terms.json (built by hpo_to_extraction_json.py from your HPO version):

  hp_trees.json     [{"code": "HP:0001396", "description": ..., "parent_branches": [[...], ...]}, ...]
                    one entry per term; each branch is one route from the term up to the
                    root, written leaf first, root last, as 7-digit codes without "HP:"
                    (the format dPhenome expects).
  id_to_term.json   {"HP:0001396": "Cholestasis", ...}

Every current term is included. Alternative (replaced) IDs get an entry too, carrying
the routes of the term that replaced them, so patient files with old codes still work.
Obsolete terms with no replacement are left out of hp_trees.json and listed.

Optional checks:
  --compare OLD_hp_trees.json   for each term in the old file, compares its set of routes
                                with the new one and reports any difference
  --check FILE [FILE ...]       symptom CSVs (first column = HPO code) or patient JSONs;
                                reports codes that are missing, retired or obsolete

Usage:
  python3.14 build_hpo_lookup_files.py -j hpo_terms.json \
      --trees-out hp_trees_full.json --map-out id_to_term_full.json \
      --compare hp_trees.json --check symptom_list_*.csv
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path


PAD = "-------"


def digits(code):
    return code.replace("HP:", "").zfill(7)


def build_routes(terms):
    """{term: [route, ...]}; each route is a tuple of 7-digit codes, leaf first, root last."""
    memo = {}

    def routes(tid):
        hit = memo.get(tid)
        if hit is not None:
            return hit
        parents = [p for p in terms[tid]["parents"] if p in terms]
        me = digits(tid)
        if not parents:
            out = [(me,)]
        else:
            out = []
            for p in parents:
                for r in routes(p):
                    out.append((me,) + r)
        memo[tid] = out
        return out

    sys.setrecursionlimit(max(10000, sys.getrecursionlimit()))
    for tid, t in terms.items():
        if not t["obsolete"]:
            routes(tid)
    return memo


def compare(old_path, new_entries):
    old = json.loads(Path(old_path).read_text(encoding="utf-8"))
    new = {digits(e["code"]): {tuple(b) for b in e["parent_branches"]} for e in new_entries}
    same, missing, diff = 0, [], []
    for e in old:
        code = digits(str(e.get("code", "")))
        # old files pad routes with "-------" to a common length; pairD ignores the padding
        o = {tuple(x for x in (str(y).zfill(7) for y in b) if x != PAD) for b in e.get("parent_branches", [])}
        n = new.get(code)
        if n is None:
            missing.append(code)
        elif n == o:
            same += 1
        else:
            diff.append((code, o, n))
    print(f"\nComparison with {old_path} ({len(old)} terms):")
    print(f"  identical routes : {same}")
    print(f"  different routes : {len(diff)}")
    print(f"  (routes compared without the '{PAD}' padding of the old file)")
    print(f"  not in new file  : {len(missing)}" + (f"  e.g. {', '.join(missing[:10])}" if missing else ""))
    for code, o, n in diff[:10]:
        print(f"  HP:{code}: old {len(o)} route(s), new {len(n)}; "
              f"only old: {len(o - n)}, only new: {len(n - o)}")
        for r in sorted(o - n)[:2]:
            print(f"      only old: {' > '.join(r)}")
        for r in sorted(n - o)[:2]:
            print(f"      only new: {' > '.join(r)}")
    if len(diff) > 10:
        print(f"  ... {len(diff) - 10} more")
    return not diff and not missing


def codes_in(path):
    """HPO codes in a symptom CSV (first column) or anywhere in a JSON file."""
    text = Path(path).read_text(encoding="utf-8")
    if path.lower().endswith(".csv"):
        rows = list(csv.reader(text.splitlines()))
        return [r[0].strip() for r in rows[1:] if r and re.fullmatch(r"HP:\d{7}", r[0].strip())]
    return [f"HP:{m}" for m in re.findall(r"(?:HP:)?(\d{7})", text)]


def check(paths, terms, alt):
    print("\nCode check:")
    ok = True
    for p in paths:
        codes = codes_in(p)
        bad = []
        for c in dict.fromkeys(codes):
            if c in terms and not terms[c]["obsolete"]:
                continue
            if c in alt:
                bad.append(f"{c} retired -> {alt[c]} {terms[alt[c]]['name']} (kept as alias)")
            elif c in terms:
                bad.append(f"{c} {terms[c]['name']}: OBSOLETE, no replacement (not in hp_trees)")
            else:
                bad.append(f"{c}: not in HPO (not in hp_trees)")
        print(f"  {p}: {len(set(codes))} codes" + ("" if bad else ", all current"))
        for b in bad:
            print(f"      {b}")
        ok = ok and not bad
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-j", "--hpo-terms", default="hpo_terms.json")
    ap.add_argument("--trees-out", default="hp_trees_full.json")
    ap.add_argument("--map-out", default="id_to_term_full.json")
    ap.add_argument("--compare", help="existing hp_trees.json to compare with")
    ap.add_argument("--check", nargs="+", default=[], help="symptom CSVs or patient JSONs to check")
    args = ap.parse_args()

    d = json.loads(Path(args.hpo_terms).read_text(encoding="utf-8"))
    terms, alt = d["terms"], d.get("alt_id_map", {})
    meta = d.get("meta", {})
    print(f"Source: {args.hpo_terms} ({meta.get('source_file', '?')}, {meta.get('data_version', '?')})")

    memo = build_routes(terms)
    entries = []
    for tid in sorted(memo):
        entries.append({"code": tid, "description": terms[tid]["name"],
                        "parent_branches": [list(r) for r in memo[tid]]})
    n_alias = 0
    for old_id, new_id in sorted(alt.items()):
        if new_id in memo and old_id not in memo:
            entries.append({"code": old_id, "description": terms[new_id]["name"], "replaced_by": new_id,
                            "parent_branches": [list(r) for r in memo[new_id]]})
            n_alias += 1
    obsolete_dropped = [t for t, v in terms.items() if v["obsolete"] and t not in alt]

    id_map = {tid: terms[tid]["name"] for tid in memo}
    for old_id, new_id in alt.items():
        if new_id in terms:
            id_map.setdefault(old_id, terms[new_id]["name"])

    Path(args.trees_out).write_text(json.dumps(entries, separators=(",", ":")), encoding="utf-8")
    Path(args.map_out).write_text(json.dumps(id_map, ensure_ascii=False, indent=0), encoding="utf-8")

    n_routes = sum(len(memo[t]) for t in memo)
    roots = sorted({r[-1] for t in memo for r in memo[t]})
    print(f"Wrote {args.trees_out}: {len(memo)} current terms + {n_alias} retired-ID aliases, "
          f"{n_routes} routes, root(s): {', '.join(roots)}")
    print(f"Wrote {args.map_out}: {len(id_map)} IDs")
    print(f"Obsolete terms without replacement left out: {len(obsolete_dropped)}")

    ok = True
    if args.compare:
        ok = compare(args.compare, entries) and ok
    if args.check:
        ok = check(args.check, terms, alt) and ok
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
