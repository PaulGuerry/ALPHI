#!/usr/local/bin/python3.14
"""Parse the disease tables (S2-S45) of the Phenomizer SI (Kohler et al. 2009, AJHG) into
symptom lists for generate_synthetic_patients.py.

Usage: parse_phenomizer_si.py SI.txt hp.obo OUTDIR
  SI.txt  output of `pdftotext -layout` on the SI PDF
  hp.obo  current HPO release (obsolete IDs are mapped through replaced_by / alt_id)
Writes OUTDIR/symptom_list_<slug>.csv (code,label,freq,sex; sex = M or F for the SI's
"Male:"/"Female:" rows, else empty), OUTDIR/phenomizer_diseases.csv
(one row per disease) and OUTDIR/phenomizer_all_terms.csv (every row, with the raw text and
how it was resolved).
Rules: textual frequencies -> Table S1 values; "A and B" -> both terms at the row frequency;
IDs in MANUAL and CORRECTIONS replaced first; "A or B" and "A and/or B" -> first term only; a term appearing twice in one disease keeps its highest frequency.
"""
import re, sys, csv, os, unicodedata
from fractions import Fraction

WORDS = {"very rare": .01, "rare": .05, "occasional": .075, "main": .25, "frequent": .33,
         "typical": .5, "common": .5, "hallmark": .9}

def load_hpo(path):
    name, alt, repl, obs = {}, {}, {}, set()
    cur = None
    for line in open(path):
        line = line.rstrip("\n")
        if line == "[Term]": cur = None; continue
        if line.startswith("id: HP:"): cur = line[4:]
        elif cur and line.startswith("name: "): name[cur] = line[6:]
        elif cur and line.startswith("alt_id: "): alt[line[8:]] = cur
        elif cur and line.startswith("replaced_by: "): repl[cur] = line[13:]
        elif cur and line == "is_obsolete: true": obs.add(cur)
    return name, alt, repl, obs

def resolve(t, H):
    name, alt, repl, obs = H
    if t in alt: return alt[t], "alt_id"
    if t in obs: return (repl[t], "replaced_by") if t in repl else (None, "obsolete, no replacement")
    if t in name: return t, "current"
    return None, "unknown"

# SI IDs absent from or reused in current HPO, mapped by the SI label
MANUAL = {"HP:0001318": "HP:0001252",                      # "Hypotonia": ID not in HPO
          "HP:0005833": "HP:0011304 and HP:0010055"}       # "Broad thumbs and broad great toes": ID now an obsolete joint term

# SI rows whose ID points to an unrelated term in current HPO (checked against the SI label):
# (table, SI ID) -> corrected ID, with the SI label for reference
CORRECTIONS = {
    ("S3",  "HP:0004394"): ("HP:0001410", "Liver dysfunction"),
    ("S6",  "HP:0000945"): ("HP:0000954", "Transverse palmar crease"),
    ("S7",  "HP:0000799"): ("HP:0007925", "Lacrimal duct aplasia or stenosis"),
    ("S9",  "HP:0000262"): ("HP:0002623", "Overriding aorta"),
    ("S9",  "HP:0002722"): ("HP:0001373", "Joint dislocations"),
    ("S11", "HP:0000688"): ("HP:0000668", "Hypodontia"),
    ("S13", "HP:0001182"): ("HP:0001882", "Leukopenia"),
    ("S14", "HP:0001120"): ("HP:0009465", "Ulnar deviation of fingers"),
    ("S21", "HP:0005352"): ("HP:0000599", "Unusual hairline with hair growth on temples extending to lateral eyebrow"),
    ("S24", "HP:0003190"): ("HP:0011803", "Bifid nose"),
    ("S25", "HP:0001495"): ("HP:0001459", "I-III toe syndactyly"),
    ("S26", "HP:0002253"): ("HP:0002353", "EEG abnormalities"),
    ("S29", "HP:0000974"): ("HP:0009741", "Nephrosclerosis"),
    ("S29", "HP:0000629"): ("HP:0000670", "Severe dental caries"),
    ("S30", "HP:0003720"): ("HP:0003270", "Abdominal distension"),
    ("S33", "HP:0002857"): ("HP:0002859", "Rhabdomyosarcoma"),
    ("S35", "HP:0000236"): ("HP:0000238", "Hydrocephalus"),
    ("S37", "HP:0001135"): ("HP:0001212", "Persistent fetal fingertip pads"),
    ("S43", "HP:0000717"): ("HP:0009717", "Cortical tubers"),
    ("S43", "HP:0009717"): ("HP:0009718", "Subependymal giant-cell astrocytoma"),
    ("S43", "HP:0009554"): ("HP:0009594", "Retinal hamartoma"),
    ("S45", "HP:0006891"): ("HP:0001331", "Absent septum pellucidum"),
    ("S45", "HP:0002189"): ("HP:0002119", "Ventriculomegaly"),
}

def freq(s):
    s = s.strip().lower()
    if m := re.fullmatch(r"([\d.]+)\s*%", s): return float(m[1]) / 100
    if m := re.fullmatch(r"(\d+)\s*/\s*(\d+)", s): return int(m[1]) / int(m[2])
    if s in WORDS: return WORDS[s]
    return None

FREQ_END = re.compile(r"\s{2,}(?:and/or\s+)?(?P<freq>[\d.]+\s*%|\d+\s*/\s*\d+|very rare|rare|occasional|main|frequent|typical|common|hallmark)\s*$", re.I)
IDS = re.compile(r"(?:(?:and/or|and|or)\s+)?HP:\d{7}(?:\s+(?:and/or|and|or))?")
CAP = re.compile(r"Table (S(?:[2-9]|[1-4]\d)): ([^\[.]+?)(?:\s*\[MIM\s*(\d+)\])?\.")

def main(si, obo, out):
    H = load_hpo(obo)
    text = open(si).read()
    os.makedirs(out, exist_ok=True)
    # each disease table precedes its caption: split the text at the captions
    caps = list(CAP.finditer(text))
    allrows, dis = [], []
    start = text.find("Table S1:")
    start = text.find("\n", text.find("values given in this table"))
    for c in caps:
        block = text[start:c.start()]; start = c.end()
        tab, title, mim = c[1], " ".join(c[2].split()), c[3]
        title = re.sub(r"\s*\[MIM\s*\d*$", "", title)
        rows = []
        block = block.replace("HP:000032 and", "HP:0000327 and")   # truncated ID in S37 (Hypoplastic maxilla)
        for line in block.splitlines():
            if "HP:" not in line: continue
            fm = FREQ_END.search(line)
            body = line[:fm.start()] if fm else line
            ids = " ".join(IDS.findall(body))
            label = " ".join(IDS.sub(" ", body).split())
            if fm:                                            # a new row (frequency on its line)
                rows.append([label, ids, fm["freq"]])
            elif rows:                                        # wrapped line: more IDs for the row above
                rows[-1][1] += " " + ids
                if label: rows[-1][0] += " " + label
        terms, sex_of = {}, {}
        for label, ids, fr in rows:
            sm = re.match(r"\s*(male|female)s?\s*:", label, re.I)
            row_sex = sm[1][0].upper() if sm else ""
            ids_raw = ids
            for old, new in MANUAL.items(): ids = ids.replace(old, new)
            ids = re.sub(r"HP:\d{7}", lambda m: CORRECTIONS.get((tab, m[0]), (m[0],))[0], ids)
            f = freq(fr)
            parts = [("", t) for t in re.findall(r"HP:\d{7}", ids)]
            joiner = "or" if re.search(r"\bor\b", ids) else "and"   # "or" and "and/or": first term only
            use = [parts[0][1]] if joiner == "or" else [p[1] for p in parts]
            for t in [p[1] for p in parts]:
                new, how = resolve(t, H)
                kept = t in use and new is not None and f is not None
                allrows.append(dict(table=tab, disease=title, mim=mim or "", label_si=label, ids_si=ids_raw, ids_used=ids,
                                    freq_si=fr, freq=f, term_in=t, term=new or "", hpo_name=H[0].get(new, ""),
                                    resolution=how, combined=joiner if len(parts) > 1 else "", sex=row_sex,
                                    kept=kept))
                if kept:
                    terms[new] = max(terms.get(new, 0), f)
                    # a term listed both as sex-specific and not keeps the non-specific entry
                    sex_of[new] = row_sex if sex_of.get(new, row_sex) == row_sex else ""
        slug = re.sub(r"[^a-z0-9]+", "_", unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode().lower().split("(")[0]).strip("_")
        fn = f"symptom_list_{slug}.csv"
        with open(os.path.join(out, fn), "w", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n"); w.writerow(["code", "label", "freq", "sex"])
            for t, f in sorted(terms.items(), key=lambda x: -x[1]):
                w.writerow([t, H[0][t], f"{f:.3f}", sex_of[t]])
        dis.append(dict(table=tab, disease=title, mim=mim or "", file=fn, n_rows=len(rows), n_terms=len(terms),
                        n_freq_ge_0_5=sum(f >= .5 for f in terms.values()),
                        mean_terms_per_patient=round(sum(terms.values()), 1)))
    for name, rows in [("phenomizer_diseases.csv", dis), ("phenomizer_all_terms.csv", allrows)]:
        with open(os.path.join(out, name), "w", newline="") as fh:
            w = csv.DictWriter(fh, rows[0].keys(), lineterminator="\n"); w.writeheader(); w.writerows(rows)

if __name__ == "__main__":
    main(*sys.argv[1:4])
