# ALPHI in the browser

A static web page that runs ALPHI entirely in the user's browser, with
[Pyodide](https://pyodide.org) (CPython compiled to WebAssembly). The patient data never leave
the browser tab, and the page can be hosted on any static host (GitHub Pages, Netlify, a
university web space): there is no server-side code.

## How it works

The pipeline is the repository's own `alphi.py`, `pgmodules.py`, `latent_classes.py`
and `term_selection.py`, copied unchanged into `web/core` by `build.sh`. Only the
dependencies that do not exist in Pyodide are replaced:

| Dependency | Why it is missing | Replacement |
|---|---|---|
| umap-learn | needs numba (LLVM) | `py/umap_lite.py`: port of umap-learn 0.5 for precomputed distances (exact kNN, fuzzy simplicial set, spectral initialisation with multi-component layout); the SGD layout optimisation runs in JavaScript (`js/umap_sgd.js`) |
| hdbscan | Cython extension | `py/hdbscan/`: scikit-learn's `HDBSCAN` (same algorithm) and a port of `hdbscan.validity` (DBCV) |
| lifelines | pulls in autograd, formulaic, matplotlib | `py/shims/lifelines`: Kaplan–Meier, log-rank tests, Cox model (Efron ties) |
| orjson, tqdm | compiled / not needed | `py/shims` |
| multiprocessing | no processes in a browser | serial pool (`py/alphi_runner.py`) |

numpy, scipy, scikit-learn and pandas come from Pyodide itself.

`umap_lite` is not bit-identical to umap-learn (different random streams), so it was checked
on the results instead. On the MYO5B cohort (137 patients, `--min-symptoms 8 --stability 50
--lca`), the same distance matrix gave the same final ALPHI groups as the published run
(69, 43 and 25 patients, adjusted Rand index 1.0 against the 11 November 2026 run made with
umap-learn and hdbscan).

## Run it locally

```bash
bash web/build.sh            # copies the pipeline and the HPO files into web/core
cd web && python3 -m http.server 8000
```

Then open <http://localhost:8000>. The first run downloads Pyodide and its packages
(about 40 MB) and the HPO tree (15 MB); the browser caches them afterwards.

## Deploy

Run `build.sh`, then publish the `web` folder as it is (for GitHub Pages: Settings → Pages,
with the folder in a branch, or a workflow that runs `build.sh` and uploads `web`).

## Input

- JSON in the PytheasDB / ALPHI format: a list of patients with `id`, `gene` and
  `symptoms` (HPO codes, or `[code, …]` lists as in `261013_pytheasDB.json`); optional
  `alivedead`, `alivedeadage` and `doi` are used as in the command-line version.
- CSV or TSV: columns `id` and `hpo` (HPO codes separated by `;`), optionally `gene`,
  `status` (alive/dead) and `age`. Without a gene column, all patients form one cohort.

## Tests

- `tests/test_myo5b_cluster.py`: clustering step on a reference distance matrix (`cdm.csv`).
- The whole pipeline can be run in CPython with the replacements, which is how the MYO5B
  check above was made:

```bash
PYTHONPATH=web/py/shims:web/py:web/core python3 web/core/alphi.py \
  -f 261013_pytheasDB.json --gene MYO5B --min-symptoms 8 --stability 50 --lca
```
