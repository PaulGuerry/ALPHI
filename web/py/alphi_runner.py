"""Runs the ALPHI pipeline (core/alphi.py, unchanged) in the browser through Pyodide.

The pipeline script is the same file as in the repository. Only its dependencies are
replaced: umap-learn by umap_lite (numba-free, layout optimisation in JavaScript), hdbscan
by scikit-learn's HDBSCAN plus a port of hdbscan.validity, lifelines/orjson/tqdm by small
stand-ins (py/shims), and multiprocessing by a serial pool (no processes in a browser).
"""
import contextlib, glob, io, json, os, runpy, sys, time, zipfile
import numpy as np
import pandas as pd

CORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core")


class _SerialPool:
    """multiprocessing.Pool replacement: same imap/map interface, run in this thread."""
    def __init__(self, processes=None, *a, **k):
        pass

    def imap(self, f, it, chunksize=1):
        return map(f, it)

    def map(self, f, it, chunksize=None):
        return list(map(f, it))

    def starmap(self, f, it, chunksize=None):
        return [f(*a) for a in it]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def close(self):
        pass

    def join(self):
        pass


def _patch_environment():
    import multiprocessing
    if sys.platform == "emscripten":
        multiprocessing.Pool = _SerialPool
        multiprocessing.cpu_count = lambda: 1
    try:
        import tabulate  # noqa: F401
    except ImportError:
        import tabulate_fallback
        sys.modules["tabulate"] = tabulate_fallback


def _display_embedding(folder, n_neighbors):
    """2-D UMAP of the distance matrix, for display only (clusters come from the pipeline)."""
    from scipy.spatial.distance import squareform
    from umap_lite import UMAP
    cdm = pd.read_csv(os.path.join(folder, "cdm.csv"))
    D = squareform(cdm["d"].to_numpy())
    nn = max(2, min(n_neighbors, D.shape[0] - 1))
    return UMAP(n_components=2, metric="precomputed", n_neighbors=nn, min_dist=0.1, random_state=25).fit_transform(D)


def _summarise(clusters, folder, top=8):
    """Per cluster: size, stability (if computed) and the terms that most distinguish it
    (prevalence in the cluster minus prevalence in the rest of the cohort)."""
    labels = json.load(open(os.path.join(CORE, "id_to_term.json"), encoding="utf-8"))
    opt = clusters["clustering_results"]["optimal"]
    sets = {c["cluster_id"]: [set(p["symptoms"]) for p in c["patients"]] for c in opt["clusters"]}
    n_all = sum(len(v) for v in sets.values())
    out = []
    for c in opt["clusters"]:
        cid = c["cluster_id"]
        mine = sets[cid]
        rest = [s for k, v in sets.items() if k != cid for s in v]
        codes = set().union(*mine)
        rows = []
        for code in codes:
            p_in = sum(code in s for s in mine) / len(mine)
            p_out = sum(code in s for s in rest) / len(rest) if rest else 0.0
            rows.append((p_in - p_out, p_in, p_out, code))
        rows.sort(reverse=True)
        out.append({"cluster": cid, "n": len(mine),
                    "stability": c.get("stability_jaccard_mean"),
                    "few_feature_flag": c.get("flag_few_features", False),
                    "terms": [{"code": "HP:" + code, "label": labels.get("HP:" + code, ""),
                               "in": round(p_in * 100), "rest": round(p_out * 100)}
                              for _, p_in, p_out, code in rows[:top]]})
    lca = {}
    lp = os.path.join(folder, "lca_check.json")
    if os.path.exists(lp):
        l = json.load(open(lp))
        lca = {k: l.get(k) for k in ("k_alphi", "k_lca", "ari_alphi_lca", "decision", "adopted", "lca_valid")}
    return {"n_patients": n_all, "k": opt["num_clusters_k"], "clusters": out, "lca": lca,
            "column": opt["clustering_column"]}


def run(options_json):
    """options: patient_file, genes, min_symptoms, stability, lca, excluded_doi,
    exclude_uncertain, digestive, workdir. Returns a JSON string with the summary, the
    display coordinates and the path of a zip of all outputs."""
    o = json.loads(options_json)
    _patch_environment()
    work = o.get("workdir", "/work")
    os.makedirs(work, exist_ok=True)
    os.chdir(work)
    for old in glob.glob(os.path.join(work, "[0-9]" * 6 + "_*")):
        import shutil
        shutil.rmtree(old, ignore_errors=True)
    argv = ["alphi.py", "-f", o["patient_file"], "--gene", *o["genes"],
            "--min-symptoms", str(int(o.get("min_symptoms", 5))),
            "--stability", str(int(o.get("stability", 50))),
            "--excluded-doi", o.get("excluded_doi") or "",
            "--output-json", "clusters.json", "--workers", "1",
            "--trees-file", os.path.join(CORE, "hp_trees.json")]
    if o.get("lca", True):
        argv.append("--lca")
    if o.get("exclude_uncertain"):
        argv.append("--exclude-uncertain-genetics")
    if o.get("digestive"):
        argv.append("--digestive")
    sys.argv = argv
    t0 = time.time()
    stdout, stderr = sys.stdout, sys.stderr
    try:
        runpy.run_path(os.path.join(CORE, "alphi.py"), run_name="__main__")
    except SystemExit as e:
        if e.code not in (0, None):
            raise RuntimeError(f"pipeline stopped (exit code {e.code}); see the log")
    finally:
        sys.stdout, sys.stderr = stdout, stderr          # the pipeline's TeeLogger replaces them
    folders = sorted(glob.glob(os.path.join(work, "[0-9]" * 6 + "_*")), key=os.path.getmtime)
    if not folders:
        raise RuntimeError("no output folder: the cohort may be empty (check the gene and minimum symptoms)")
    folder = folders[-1]
    clusters = json.load(open(os.path.join(work, "clusters.json"), encoding="utf-8"))
    umap_csv = pd.read_csv(os.path.join(folder, "pairD_UMAP.csv"))
    from pairD_dyn import n_neighbors_for
    xy = _display_embedding(folder, n_neighbors_for(len(umap_csv)))
    label_col = clusters["clustering_results"]["optimal"]["clustering_column"]
    points = [{"id": i, "x": float(a), "y": float(b), "cluster": int(c)}
              for i, a, b, c in zip(umap_csv["ids"], xy[:, 0], xy[:, 1], umap_csv[label_col])]
    summary = _summarise(clusters, folder)
    zpath = os.path.join(work, "alphi_results.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(glob.glob(os.path.join(folder, "*"))):
            if not f.endswith(".py"):
                z.write(f, os.path.join(os.path.basename(folder), os.path.basename(f)))
        z.write(os.path.join(work, "clusters.json"), "clusters.json")
    return json.dumps({"seconds": time.time() - t0, "folder": os.path.basename(folder), "zip": zpath,
                       "points": points, "summary": summary}, default=lambda v: v.item() if hasattr(v, "item") else str(v))
