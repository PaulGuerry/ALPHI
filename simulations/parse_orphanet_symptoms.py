#!/usr/local/bin/python3.14
"""
Convert an Orphanet symptom list (copied from the Orphanet page, one frequency
header followed by "Label HP:0000000" lines) into an HPO frequency CSV.

Each symptom gets a numeric frequency drawn uniformly within its Orphanet class.
Use --seed so the same input always gives the same frequencies.

Changes from the previous version:
  - "Very rare" (Orphanet's name for the 1-4 % class) is recognised. Before, it was
    not, so very rare symptoms silently inherited the previous class (occasional).
  - "Obligate" is accepted as a synonym of "always present"; "Excluded" symptoms
    (0 %) are left out.
  - Unknown header lines and symptoms appearing before any header are reported on
    stderr instead of being written with a frequency of 0.00.
  - A symptom listed twice keeps its first entry (warning on stderr).
  - Optional --seed for reproducible frequencies, and -o to write a file directly.
"""
import argparse
import csv
import random
import re
import sys

FREQ_RANGES = {   # Orphanet class -> (min, max)
    "always present": (1.00, 1.00),
    "obligate": (1.00, 1.00),
    "very frequent": (0.80, 0.99),
    "frequent": (0.30, 0.79),
    "occasional": (0.05, 0.29),
    "very rare": (0.01, 0.04),
    "rare": (0.01, 0.04),
    "excluded": None,          # 0 %: symptom is left out
}

SYMPTOM = re.compile(r'^(.*?)\s+(HP:\d{7})$')


def header_class(line):
    """Return the frequency class named by a header line, or None if it is not a header."""
    key = re.sub(r'\s*\(.*\)\s*$', '', line).strip().lower()   # drops e.g. " (99-80%)"
    return key if key in FREQ_RANGES else None


def parse_symptoms(file_path, rng):
    current = None
    rows, seen = [], {}
    with open(file_path, encoding='utf-8') as f:
        for n, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            cls = header_class(line)
            if cls is not None:
                current = cls
                continue
            m = SYMPTOM.match(line)
            if not m:
                print(f"WARNING {file_path}:{n}: unrecognised line skipped: {line!r}", file=sys.stderr)
                continue
            label, code = m.group(1).strip(), m.group(2)
            if current is None:
                print(f"WARNING {file_path}:{n}: symptom before any frequency header skipped: {line!r}",
                      file=sys.stderr)
                continue
            if code in seen:
                print(f"WARNING {file_path}:{n}: {code} {label} already listed as {seen[code]}; "
                      f"keeping the first entry", file=sys.stderr)
                continue
            bounds = FREQ_RANGES[current]
            if bounds is None:                     # excluded
                continue
            low, high = bounds
            val = low if low == high else rng.uniform(low, high)
            seen[code] = current
            rows.append((code, label, f"{val:.2f}", current))
    return rows


def main():
    ap = argparse.ArgumentParser(description="Parse an Orphanet symptom list into an HPO frequency CSV.")
    ap.add_argument('-f', '--input-file', required=True, help="Orphanet symptom list (text).")
    ap.add_argument('-o', '--output', help="Output CSV (default: standard output).")
    ap.add_argument('--seed', type=int, help="Random seed, for reproducible frequencies.")
    ap.add_argument('--with-class', action='store_true',
                    help="Add a fourth column with the Orphanet frequency class.")
    args = ap.parse_args()

    rows = parse_symptoms(args.input_file, random.Random(args.seed))

    out = open(args.output, 'w', newline='', encoding='utf-8') if args.output else sys.stdout
    w = csv.writer(out, quoting=csv.QUOTE_MINIMAL)
    header = ["HPO Code", "HPO Label or Descriptor", "Frequency"] + (["Orphanet Class"] if args.with_class else [])
    w.writerow(header)
    for code, label, freq, cls in rows:
        w.writerow([code, label, freq] + ([cls] if args.with_class else []))
    if args.output:
        out.close()


if __name__ == '__main__':
    main()
