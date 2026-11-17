#!/bin/bash
# Copies the ALPHI pipeline and its data from the repository root into web/core, unchanged.
# Run from the repository root:  bash web/build.sh
set -e
cd "$(dirname "$0")"
mkdir -p core
for f in alphi.py pgmodules.py latent_classes.py term_selection.py hp_trees.json id_to_term.json; do
  cp "../$f" core/
done
# manifest of the Python files the worker loads into Pyodide's file system
python3 - <<'PY'
import json, os
files = []
for root in ("py", "core"):
    for d, _, fs in os.walk(root):
        files += [os.path.join(d, f).replace(os.sep, "/") for f in fs if f.endswith((".py", ".json"))]
json.dump(sorted(files), open("manifest.json", "w"), indent=1)
print(len(files), "files in manifest.json")
PY
