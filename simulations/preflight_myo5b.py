#!/usr/local/bin/python3.14
"""
Checks, without clustering anything, that everything is in place for the three MYO5B runs
of pairD_phenome.py (OB, Jaccard, Resnik):

  1. pairD_phenome.py on PATH, and the pgmodules.py next to it has the distance switch
  2. hp_trees.json (complete), id_to_term.json and, for Resnik, phenotype.hpoa
  3. the run folders: pairD_phenome.py uses hp_trees.json and id_to_term.json from the
     current folder if present there, else the ones next to pairD_phenome.py. Reports which
     copy each run folder will use, and flags a local copy that differs from the main one
     (it would be used silently). A missing id_to_term.json only gives a warning but
     changes the OB distance (opposite-direction terms).
  4. the patient file: patients selected as pairD_phenome.py does (gene, --min-symptoms,
     --exclude-uncertain-genetics, the excluded DOI)
  5. for each run setting, in a fresh Python process (environment variables are read when
     pgmodules.py is imported): the setting actually in effect, and the distance between
     a few real patient pairs, so the three settings can be seen to differ

Usage (in ~/myo5b_distances):
  python3.14 preflight_myo5b.py --patients ~/GITHUB/pairD_phenome/261013_patientData.json
  python3.14 preflight_myo5b.py --patients ... --exclude-uncertain-genetics --min-symptoms 5
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

RUNS = {
    "ob_nearest": {"PGM_DISTANCE": "dphenome", "PGM_ORGAN_MODE": "nearest"},
    "jaccard": {"PGM_DISTANCE": "jaccard"},
    "resnik": {"PGM_DISTANCE": "resnik"},
}
DEFAULT_EXCLUDED_DOI = "https://doi.org/10.1136/adc.43.232.646"   # pairD_phenome.py's default

CHILD = r"""
import json, os, sys
pdir, trees, idmap, pairs = sys.argv[1], sys.argv[2], sys.argv[3], json.loads(sys.argv[4])
sys.path.insert(0, pdir)
import pgmodules as pgm
tree = {str(b.get("code", "")).replace("HP:", "").zfill(7): b for b in json.load(open(trees, "rb"))}
m = json.load(open(idmap, encoding="utf-8")) if os.path.exists(idmap) else None
info = {"file": pgm.__file__, "distance": getattr(pgm, "DISTANCE_MODE", None),
        "organ_mode": getattr(pgm, "ORGAN_MODE", None)}
if info["distance"] == "resnik":
    info["resnik"] = pgm.resnik_ic_description()
info["d"] = [round(pgm.dPhenome(a, b, tree, digestive_only=False, id_to_term_map=m), 4) for a, b in pairs]
print(json.dumps(info))
"""


def ok(msg):
    print(f"  [ok]   {msg}")


def bad(msg, problems):
    print(f"  [FAIL] {msg}")
    problems.append(msg)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patients", required=True)
    ap.add_argument("--gene", default="MYO5B")
    ap.add_argument("--min-symptoms", type=int, default=5)
    ap.add_argument("--exclude-uncertain-genetics", action="store_true")
    ap.add_argument("--excluded-doi", default=DEFAULT_EXCLUDED_DOI)
    ap.add_argument("--base", default=".", help="folder holding the run folders (default: here)")
    args = ap.parse_args()
    problems = []

    print("1. pairD_phenome.py and pgmodules.py")
    exe = shutil.which("pairD_phenome.py")
    if not exe:
        bad("pairD_phenome.py not found on PATH", problems)
        sys.exit("\nCannot continue.")
    pdir = Path(os.path.realpath(exe)).parent
    ok(f"pairD_phenome.py: {os.path.realpath(exe)}")
    pgm_src = (pdir / "pgmodules.py").read_text(encoding="utf-8") if (pdir / "pgmodules.py").exists() else ""
    if "PGM_DISTANCE" in pgm_src and "DISTANCE_MODES" in pgm_src:
        ok("pgmodules.py has the distance switch (PGM_DISTANCE)")
    else:
        bad(f"pgmodules.py in {pdir} has no distance switch: install the updated one", problems)

    print("\n2. Lookup files next to pairD_phenome.py")
    trees, idmap, hpoa = pdir / "hp_trees.json", pdir / "id_to_term.json", pdir / "phenotype.hpoa"
    if trees.exists():
        n = len(json.load(open(trees, "rb")))
        (ok if n >= 15000 else lambda m: bad(m, problems))(f"hp_trees.json: {n} entries"
                                                           + ("" if n >= 15000 else " (incomplete: rebuild it)"))
    else:
        bad("hp_trees.json missing", problems)
    (ok if idmap.exists() else lambda m: bad(m, problems))(f"id_to_term.json {'present' if idmap.exists() else 'missing'}")
    (ok if hpoa.exists() else lambda m: bad(m, problems))(f"phenotype.hpoa {'present' if hpoa.exists() else 'missing (needed for Resnik)'}")

    print("\n3. Run folders (current-folder copy first, else the one next to pairD_phenome.py)")
    for run in RUNS:
        folder = Path(args.base) / run
        if not folder.exists():
            print(f"  [--]   {folder}/ not created yet (mkdir only: no copies or links needed)")
            continue
        for f in ("hp_trees.json", "id_to_term.json"):
            local, main = folder / f, pdir / f
            if local.exists():
                same = main.exists() and os.path.realpath(local) == os.path.realpath(main) or \
                    (main.exists() and local.read_bytes() == main.read_bytes())
                (ok if same else lambda m: bad(m, problems))(
                    f"{folder}/{f}: local copy used" + ("" if same else f", DIFFERENT from {main}: remove it"))
            elif main.exists():
                ok(f"{folder}/{f}: uses {main}")
            else:
                bad(f"{folder}/{f}: not found in the folder nor next to pairD_phenome.py", problems)

    print("\n4. Patient file")
    db = json.load(open(args.patients, encoding="utf-8"))
    db = db["patients"] if isinstance(db, dict) and "patients" in db else db
    sel, multi = [], 0
    for p in db:
        genes = p.get("gene") or []
        genes = genes if isinstance(genes, list) else [g.strip() for g in str(genes).replace(";", ",").split(",")]
        if args.gene not in genes:
            continue
        # pairD_phenome.py checks the patient's own "doi" field (not articles[].doi)
        own = p.get("doi", [])
        own = own if isinstance(own, list) else [own]
        if args.excluded_doi and args.excluded_doi in own:
            continue
        if len(genes) > 1:
            multi += 1
            if args.exclude_uncertain_genetics:
                continue
        if len({str(s[0] if isinstance(s, list) else s) for s in p.get("symptoms") or []}) >= args.min_symptoms:
            sel.append(p)
    ok(f"{len(sel)} {args.gene} patients selected (>= {args.min_symptoms} symptoms"
       + (", single-gene only" if args.exclude_uncertain_genetics else "") + f"); {multi} have more than one gene entry"
       + ("" if args.exclude_uncertain_genetics else " and are INCLUDED (add --exclude-uncertain-genetics to drop them)"))
    in_articles = sum(any(args.excluded_doi in str(a.get("doi", "")) for a in p.get("articles") or []
                          if isinstance(a, dict)) and "doi" not in p for p in db)
    if args.excluded_doi and in_articles:
        print(f"  [note] {in_articles} patients cite the excluded DOI only under articles[].doi, which "
              "pairD_phenome.py does not check: they are NOT excluded")
    if len(sel) < 10:
        bad("fewer than 10 patients selected", problems)

    print("\n5. Distance settings (fresh process each; distances for 3 patient pairs)")
    pairs = [(sel[0], sel[1]), (sel[0], sel[len(sel) // 2]), (sel[1], sel[-1])] if len(sel) >= 3 else []
    pairs = [[{"symptoms": a["symptoms"]}, {"symptoms": b["symptoms"]}] for a, b in pairs]
    seen = {}
    for run, env in RUNS.items():
        r = subprocess.run([sys.executable, "-c", CHILD, str(pdir), str(trees), str(idmap), json.dumps(pairs)],
                           env={**os.environ, **env}, capture_output=True, text=True)
        if r.returncode != 0:
            bad(f"{run}: " + (r.stderr.strip().splitlines() or ["error"])[-1], problems)
            continue
        info = json.loads(r.stdout.strip().splitlines()[-1])
        want_d, want_o = env["PGM_DISTANCE"], env.get("PGM_ORGAN_MODE")
        if info["distance"] != want_d or (want_o and info["organ_mode"] != want_o):
            bad(f"{run}: setting not taken into account (distance {info['distance']}, organ mode {info['organ_mode']})",
                problems)
            continue
        extra = f", {info['resnik']}" if "resnik" in info else ""
        ok(f"{run:<11} distance {info['distance']}, organ mode {info['organ_mode']}{extra}: d = {info['d']}")
        seen[run] = info["d"]

    print("\n" + ("All checks passed." if not problems else f"{len(problems)} problem(s):\n  - " + "\n  - ".join(problems)))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
