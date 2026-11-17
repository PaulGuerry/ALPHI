#!/usr/local/bin/python3.14
"""
Writes a copy of the PytheasDB patient file without the patients of a gene who were
diagnosed clinically, without genetic testing ("Not tested" in their gene list), for the
sensitivity analysis of the MYO5B clustering. Patients of other genes are kept unchanged.

Usage:
  python3.14 filter_genotyped_patients.py -f 261013_patientData.json --gene MYO5B \
      -o 261013_patientData_MYO5B_genotyped.json
Then run ALPHI on the filtered file exactly as for the main analysis, e.g.:
  pairD_phenome.py -f 261013_patientData_MYO5B_genotyped.json --gene MYO5B --min-symptoms 8 \
      --stability 50 --output-json clusters_genotyped.json
and compare with compare_clusterings.py.
"""
import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-f", "--input", required=True, help="patient JSON (list of patient records)")
    ap.add_argument("--gene", default="MYO5B")
    ap.add_argument("--min-symptoms", type=int, default=8, help="only for the counts printed (ALPHI's own filter applies)")
    ap.add_argument("-o", "--output", required=True)
    a = ap.parse_args()

    db = json.load(open(a.input, encoding="utf-8"))
    of_gene = lambda p: any(a.gene in str(g) for g in (p.get("gene") or []))
    untested = lambda p: any("Not tested" in str(g) for g in (p.get("gene") or []))
    n_sym = lambda p: len(p.get("symptoms") or [])

    removed = [p for p in db if of_gene(p) and untested(p)]
    kept = [p for p in db if not (of_gene(p) and untested(p))]
    gene_kept = [p for p in kept if of_gene(p)]
    print(f"{a.gene}: {sum(of_gene(p) for p in db)} patients; removed {len(removed)} not genetically tested "
          f"({sum(n_sym(p) >= a.min_symptoms for p in removed)} with >= {a.min_symptoms} terms); "
          f"kept {len(gene_kept)} ({sum(n_sym(p) >= a.min_symptoms for p in gene_kept)} with >= {a.min_symptoms} terms)")
    if not gene_kept:
        sys.exit(f"No genotyped {a.gene} patient left.")
    with open(a.output, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=1)
    print(f"Wrote {a.output} ({len(kept)} of {len(db)} patients)")


if __name__ == "__main__":
    main()
