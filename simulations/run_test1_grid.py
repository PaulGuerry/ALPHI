#!/usr/local/bin/python3.14
"""
TEST 1 grid search: three artificial diseases (D1S0, D2S2, D3S3) built from a fixed
symptom table (build_test1_table.py), simulated with generate_test1_patients.py and
clustered with pairD_phenome.py.

Grid (each axis can be restricted on the command line; defaults = the repeat of TEST 1):
  layout       dispersed / concentrated signature symptoms (tables from build_test1_tables.py)
  substitution p, probability that a reported term is replaced by a related HPO term
  distance     dphenome (pairD's distance); jaccard (exact-code baseline); resnik (1 - Resnik
               best-match average, the standard ontology-based phenotype similarity, with
               annotation-based IC from phenotype.hpoa by default: see pgmodules.py);
               hamming also available. Needs the pgmodules.py with the PGM_DISTANCE switch
               and, for resnik, the complete hp_trees.json.
  The generation seed does not depend on layout, substitution or distance, so those
  comparisons are made on the same underlying patients.
  With substitution p > 0 the symptom metrics (Jaccard / precision / recall, and so
  pass_all) compare exact codes and are lowered by construction: use n_clusters,
  k_correct and ari. Extra iterations are then decided on k_correct.
  disease    D1S0 (no subgroups), D2S2 (two), D3S3 (three)
  N          cohort size
  k          signature symptoms per subgroup (how many features differ between subgroups)
  contrast   how specific each distinguishing feature is:
             strong = signature in 80 % of its own subgroup, 5 % of the others
             weak   = 60 % / 15 %
  reporting  r, how completely the distinguishing features are reported: each signature
             symptom a patient has is reported with probability r (1, 0.75, 0.5), so the
             observed frequencies are r x high and r x low. Core symptoms are always reported.
For D1S0, k, contrast and r set how many signature symptoms appear and at what flat
frequency (r x the D3S3 average by default), so D1S0 is matched to D3S3 at each grid point.

Self-contained: needs only generate_test1_patients.py and test1_table.csv next to it,
and pairD_phenome.py on PATH (with its pgmodules.py, hp_trees.json and id_to_term.json).
The evaluation is the same as in run_gridsearch.py (copied here, unchanged): the
optimal clustering only is scored (Jaccard / precision / recall of signature symptoms,
size error on assigned patients), with adaptive iterations and the same pass thresholds.
Added per iteration:
  n_clusters   number of non-noise clusters found
  k_correct    1 if n_clusters equals the true number of subgroups
  ari          adjusted Rand index between true subgroups and clusters, on the patients
               assigned to a cluster (noise not counted, as for the size error)

Each iteration uses its own generation seed, derived from --base-seed and the grid
point, so the whole grid is reproducible.

Output:
  -o           one row per grid point (summary statistics, pass/fail)
  --raw        one row per iteration

Usage (in the folder with the table, generate_test1_patients.py and hpo_terms.json):
  python3.14 run_test1_grid.py -t test1_table_C1.csv 2>&1 | tee test1_C1.log
  python3.14 run_test1_grid.py --dry-run
  python3.14 run_test1_grid.py --disease D1S0 --k 3 6 --contrast strong --reporting 1   # part of the grid
"""
import argparse
import shlex
import csv
import importlib.util
import json
import math
import multiprocessing as mp
import os
import shutil
import sys
import time
import traceback
import zlib
from pathlib import Path

import numpy as np

import generate_test1_patients as gt

sys.stdout.reconfigure(line_buffering=True)

# =============================================================================
# Settings and evaluation shared with run_gridsearch.py (copied, unchanged except
# where noted), so TEST 1 is scored exactly as the earlier grid search.
# =============================================================================

# pairD_phenome.py writes its results here (--output-json); deleted before each run.
CLUSTER_JSON = Path("cluster_visualization_current.json")

# Keys under which a cluster may list its member patients (first one found is used).
CLUSTER_MEMBER_KEYS = ("patients", "patient_ids", "members", "patient_list")

# Pass criterion: metric -> (threshold, "min" = must be >=, "max" = must be <=)
THRESHOLDS = {
    "avg_tier1_mean_jaccard":    (0.80, "min"),
    "avg_tier1_mean_precision":  (0.80, "min"),
    "avg_tier1_mean_recall":     (0.80, "min"),
    "avg_mean_percentage_error": (20.0, "max"),
}

N_ITER_INITIAL = 5
N_ITER_MAX = 10
CLOSE_CALL_SE = 2.0
MIN_SUCCESSFUL = 3          # fewer successful iterations than this -> status "failed"

BASE_METRICS = [
    "s2c_rmse_size", "s2c_mae_size", "s2c_mean_percentage_error", "s2c_tier1_mean_jaccard", "s2c_tier1_mean_precision", "s2c_tier1_mean_recall",
    "c2s_rmse_size", "c2s_mae_size", "c2s_mean_percentage_error", "c2s_tier1_mean_jaccard", "c2s_tier1_mean_precision", "c2s_tier1_mean_recall",
    "avg_rmse_size", "avg_mae_size", "avg_mean_percentage_error", "avg_tier1_mean_jaccard", "avg_tier1_mean_precision", "avg_tier1_mean_recall",
    "coverage",              # fraction of patients assigned to a (non-noise) cluster; not part of the pass criterion
    "size_reference_exact",  # 1 = reference sizes from cluster membership, 0 = scaled by coverage
]
STAT_SUFFIXES = ["mean", "sd", "median", "min", "max"]

MAX_CONSECUTIVE_FAILURES = 10   # stop the grid if this many iterations fail in a row
_consecutive_failures = 0
_PAIRD_MODULE = None
KEEP_RESULTS = False            # --keep-results: keep pairD_phenome.py's per-run result folders


def _nanmean(values) -> float:
    """Mean ignoring NaN (subgroups with no assigned patients); NaN if nothing is left."""
    a = np.array(values, dtype=float)
    return float(np.nanmean(a)) if a.size and np.any(~np.isnan(a)) else float("nan")


def _cluster_members(cluster: dict):
    for key in CLUSTER_MEMBER_KEYS:
        val = cluster.get(key)
        if isinstance(val, list) and val:
            ids = []
            for p in val:
                pid = (p.get("id") or p.get("patient_id")) if isinstance(p, dict) else p
                if pid is not None:
                    ids.append(str(pid))
            return ids
    return None


def assign_reference_sizes(subgroups: dict, synth_data: dict, detected_clusters: list) -> tuple[float, bool]:
    """
    Sets subgroups[key]["assigned_size"] = number of that subgroup's patients assigned to a
    non-noise cluster, so that patients left as noise are not counted as size errors.

    Exact when the clusters list their member patient IDs; otherwise each subgroup's true
    size is scaled by the overall coverage (assumes noise is spread proportionally).
    Returns (coverage, exact).
    """
    patients = synth_data.get("patients", [])
    n_total = len(patients)
    n_assigned = sum(c.get("patient_count", 0) for c in detected_clusters)
    coverage = min(1.0, n_assigned / n_total) if n_total else 1.0

    members = [_cluster_members(c) for c in detected_clusters]
    if members and all(m is not None for m in members):
        id_to_key = {str(p["id"]): f"{p['gene'][0]} (Subgroup {p['subgroup']})" for p in patients}
        counts = {}
        for m in members:
            for pid in m:
                key = id_to_key.get(pid)
                if key is not None:
                    counts[key] = counts.get(key, 0) + 1
        if counts:
            for key, sg in subgroups.items():
                sg["assigned_size"] = counts.get(key, 0)
            coverage = min(1.0, sum(counts.values()) / n_total) if n_total else 1.0
            return coverage, True

    for sg in subgroups.values():
        sg["assigned_size"] = sg["actual_size"] * coverage
    return coverage, False


def get_disease_associated_codes(g_data: dict) -> set:
    """
    Extracts disease-associated / background symptom HPO codes from parent disease metadata.
    """
    codes = set()
    for key in [
        "disease_specific_symptoms"
    ]:
        val = g_data.get(key, [])
        if val:
            for item in val:
                if isinstance(item, dict):
                    code = item.get("hpo_code") or item.get("hpo_id") or item.get("id")
                    if code:
                        codes.add(code)
                elif isinstance(item, str):
                    codes.add(item)
    #print(f"[DEBUG] get_disease_associated_codes extracted {len(codes)} codes: {list(codes)[:5]}...")
    return codes


def _get_cluster_signatures(dc: dict, k: int) -> list:
    """
    Retrieves cluster signatures from 'associated_symptoms'.
    Falls back to 'prevalent_symptoms' if k = 1.
    """
    sigs = dc.get("associated_symptoms", [])
    if not sigs and k == 1:
        sigs = dc.get("prevalent_symptoms", [])
    if not sigs and k == 1:
        raise RuntimeError("Cluster signatures ('associated_symptoms' and fallback 'prevalent_symptoms') are empty for k=1.")
    return sigs


def populate_sigs_from_prevalent_symptoms(dc: dict) -> list:
    """
    Retrieves cluster signatures from 'prevalent_symptoms' with fallback to 'associated_symptoms' for k=1.
    """
    sigs = dc.get("prevalent_symptoms", [])
    if not sigs:
        sigs = dc.get("associated_symptoms", [])
    if not sigs:
        raise RuntimeError("Cluster signatures ('prevalent_symptoms' and fallback 'associated_symptoms') are empty for k=1.")
    return sigs


def compute_cluster_signatures(detected_clusters: list, k: int) -> list:
    """
    Computes/retrieves cluster signatures across all detected clusters for k > 1.
    """
    sigs_list = []
    for dc in detected_clusters:
        sigs = _get_cluster_signatures(dc, k)
        sigs_list.append(sigs)
    return sigs_list


def create_trivial_single_cluster_matching(subgroups: dict, detected_clusters: list, eval_metadata: dict) -> dict:
    """
    Defines a trivial matching structure for k=1 so downstream metric functions 
    operate uniformly.
    """
    subgroup_keys = list(subgroups.keys())
    k = len(detected_clusters)
    best_cluster_idx = 0
    
    matching = {
        "s2c_indices": [best_cluster_idx for _ in subgroup_keys],
        "c2s_indices": [0 for _ in range(max(1, k))],
        "sim_matrix": np.ones((max(1, k), len(subgroup_keys)))
    }
    return matching


def find_optimal_cluster_matching(sigs_list: list, subgroups: dict, detected_clusters: list, eval_metadata: dict) -> dict:
    """
    Finds optimal cluster matching for k > 1 using similarity matrix calculations.
    """
    subgroup_keys = list(subgroups.keys())
    n_gen = len(subgroups)
    k = len(detected_clusters)
    
    sim_matrix = np.zeros((k, n_gen))

    for j, dc in enumerate(detected_clusters):
        sigs = sigs_list[j]
        s_det = set(
            item.get("hpo_code") or item.get("hpo_id") or item.get("id") if isinstance(item, dict) else item
            for item in sigs
        )
        
        for i, sg_key in enumerate(subgroup_keys):
            sg_info = subgroups[sg_key]
            parent_disease = sg_info["parent_disease"]
            g_data = eval_metadata.get(parent_disease, {})
            
            # Conditional background subtraction: only if n_subgroups > 1
            subgroups_dict = g_data.get("subgroups", {})
            do_subtraction = len(subgroups_dict) > 1
            disease_assoc_codes = get_disease_associated_codes(g_data) if do_subtraction else set()
            
            s_det_filtered = s_det - disease_assoc_codes
            s_gen_tier1 = set(
                item.get("hpo_code") or item.get("hpo_id") or item.get("id") if isinstance(item, dict) else item
                for item in sg_info["signature_symptoms"]
            )
            s_gen_filtered = s_gen_tier1 - disease_assoc_codes
            
            union_set = s_gen_filtered.union(s_det_filtered)
            inter_set = s_gen_filtered.intersection(s_det_filtered)
            
            j_sim = len(inter_set) / len(union_set) if len(union_set) > 0 else 0.0
            sim_matrix[j, i] = j_sim
            
    s2c_indices = [int(np.argmax(sim_matrix[:, i])) for i in range(n_gen)]
    c2s_indices = [int(np.argmax(sim_matrix[j, :])) for j in range(k)]
    
    matching = {
        "s2c_indices": s2c_indices,
        "c2s_indices": c2s_indices,
        "sim_matrix": sim_matrix
    }
    return matching


def calculate_subgroup_metrics(
    signatures: list, 
    matching: dict, 
    subgroups: dict,
    detected_clusters: list,
    eval_metadata: dict,
    k: int
) -> dict:
    """
    Unified metric calculation block running identically regardless of whether k was 1 or n.
    """
    subgroup_keys = list(subgroups.keys())
    s2c_indices = matching["s2c_indices"]
    c2s_indices = matching["c2s_indices"]

    # --- Direction 1: Subgroup -> Cluster ---
    s2c_abs_diffs = []
    s2c_pct_errors = []
    s2c_jaccards = []
    s2c_precisions = []
    s2c_recalls = []
    
    for i, sg_key in enumerate(subgroup_keys):
        sg_info = subgroups[sg_key]
        parent_disease = sg_info["parent_disease"]
        g_data = eval_metadata.get(parent_disease, {})
        
        # Conditional background subtraction: only if n_subgroups > 1
        subgroups_dict = g_data.get("subgroups", {})
        do_subtraction = len(subgroups_dict) > 1
        disease_assoc_codes = get_disease_associated_codes(g_data) if do_subtraction else set()
        
        j_best = s2c_indices[i]
        dc = detected_clusters[j_best]
        
        # Reference size = this subgroup's patients that the method assigned to a cluster;
        # patients left as noise are not counted as size errors.
        size_gen = sg_info.get("assigned_size", sg_info["actual_size"])
        size_det = dc.get("patient_count", 0)
        abs_diff = abs(size_gen - size_det)
        pct_error = (abs_diff / size_gen * 100.0) if size_gen > 0 else float("nan")
        
        s2c_abs_diffs.append(abs_diff)
        s2c_pct_errors.append(pct_error)
        
        sigs = signatures[j_best] if isinstance(signatures, list) and len(signatures) > j_best else _get_cluster_signatures(dc, k)
        
        s_det = set()
        for sig in sigs:
            if isinstance(sig, dict):
                code = sig.get("hpo_code") or sig.get("hpo_id") or sig.get("id")
                if code:
                    s_det.add(code)
            elif isinstance(sig, str):
                s_det.add(sig)
                
        s_det_filtered = s_det - disease_assoc_codes
        
        s_tier1 = set()
        for item in sg_info["signature_symptoms"]:
            if isinstance(item, dict):
                code = item.get("hpo_code") or item.get("hpo_id") or item.get("id")
                if code:
                    s_tier1.add(code)
            elif isinstance(item, str):
                s_tier1.add(item)
                
        s_tier1_filtered = s_tier1 - disease_assoc_codes
        
        t1_union = s_tier1_filtered.union(s_det_filtered)
        t1_inter = s_tier1_filtered.intersection(s_det_filtered)
        
        t1_j = len(t1_inter) / len(t1_union) if len(t1_union) > 0 else 0.0
        t1_p = len(t1_inter) / len(s_det_filtered) if len(s_det_filtered) > 0 else 0.0
        t1_r = len(t1_inter) / len(s_tier1_filtered) if len(s_tier1_filtered) > 0 else 0.0

        s2c_jaccards.append(t1_j)
        s2c_precisions.append(t1_p)
        s2c_recalls.append(t1_r)

    s2c_rmse = np.sqrt(np.mean(np.array(s2c_abs_diffs) ** 2)) if s2c_abs_diffs else 0.0
    s2c_mae = np.mean(s2c_abs_diffs) if s2c_abs_diffs else 0.0
    s2c_mean_pct = _nanmean(s2c_pct_errors)
    s2c_mean_j = np.mean(s2c_jaccards) if s2c_jaccards else 0.0
    s2c_mean_p = np.mean(s2c_precisions) if s2c_precisions else 0.0
    s2c_mean_r = np.mean(s2c_recalls) if s2c_recalls else 0.0

    # --- Direction 2: Cluster -> Subgroup ---
    c2s_abs_diffs = []
    c2s_pct_errors = []
    c2s_jaccards = []
    c2s_precisions = []
    c2s_recalls = []
    
    for j, dc in enumerate(detected_clusters):
        sigs = signatures[j] if isinstance(signatures, list) and len(signatures) > j else _get_cluster_signatures(dc, k)
        s_det = set()
        for sig in sigs:
            if isinstance(sig, dict):
                code = sig.get("hpo_code") or sig.get("hpo_id") or sig.get("id")
                if code:
                    s_det.add(code)
            elif isinstance(sig, str):
                s_det.add(sig)
                
        i_best = c2s_indices[j] if j < len(c2s_indices) else 0
        sg_key = subgroup_keys[i_best]
        sg_info = subgroups[sg_key]
        parent_disease = sg_info["parent_disease"]
        g_data = eval_metadata.get(parent_disease, {})
        
        # Conditional background subtraction: only if n_subgroups > 1
        subgroups_dict = g_data.get("subgroups", {})
        do_subtraction = len(subgroups_dict) > 1
        disease_assoc_codes = get_disease_associated_codes(g_data) if do_subtraction else set()
        
        # Reference size = this subgroup's patients that the method assigned to a cluster;
        # patients left as noise are not counted as size errors.
        size_gen = sg_info.get("assigned_size", sg_info["actual_size"])
        size_det = dc.get("patient_count", 0)
        abs_diff = abs(size_gen - size_det)
        pct_error = (abs_diff / size_gen * 100.0) if size_gen > 0 else float("nan")
        
        c2s_abs_diffs.append(abs_diff)
        c2s_pct_errors.append(pct_error)
        
        s_det_filtered = s_det - disease_assoc_codes
        s_tier1 = set()
        for item in sg_info["signature_symptoms"]:
            if isinstance(item, dict):
                code = item.get("hpo_code") or item.get("hpo_id") or item.get("id")
                if code:
                    s_tier1.add(code)
            elif isinstance(item, str):
                s_tier1.add(item)
                
        s_tier1_filtered = s_tier1 - disease_assoc_codes
        
        t1_union = s_tier1_filtered.union(s_det_filtered)
        t1_inter = s_tier1_filtered.intersection(s_det_filtered)
        
        t1_j = len(t1_inter) / len(t1_union) if len(t1_union) > 0 else 0.0
        t1_p = len(t1_inter) / len(s_det_filtered) if len(s_det_filtered) > 0 else 0.0
        t1_r = len(t1_inter) / len(s_tier1_filtered) if len(s_tier1_filtered) > 0 else 0.0
        
        c2s_jaccards.append(t1_j)
        c2s_precisions.append(t1_p)
        c2s_recalls.append(t1_r)

    c2s_rmse = np.sqrt(np.mean(np.array(c2s_abs_diffs) ** 2)) if c2s_abs_diffs else 0.0
    c2s_mae = np.mean(c2s_abs_diffs) if c2s_abs_diffs else 0.0
    c2s_mean_pct = _nanmean(c2s_pct_errors)
    c2s_mean_j = np.mean(c2s_jaccards) if c2s_jaccards else 0.0
    c2s_mean_p = np.mean(c2s_precisions) if c2s_precisions else 0.0
    c2s_mean_r = np.mean(c2s_recalls) if c2s_recalls else 0.0

    # --- Directional Average ---
    avg_rmse = (s2c_rmse + c2s_rmse) / 2.0
    avg_mae = (s2c_mae + c2s_mae) / 2.0
    avg_mean_pct = (s2c_mean_pct + c2s_mean_pct) / 2.0
    avg_mean_j = (s2c_mean_j + c2s_mean_j) / 2.0
    avg_mean_p = (s2c_mean_p + c2s_mean_p) / 2.0
    avg_mean_r = (s2c_mean_r + c2s_mean_r) / 2.0

    aggregate_metrics = {
        "s2c_rmse_size": s2c_rmse,
        "s2c_mae_size": s2c_mae,
        "s2c_mean_percentage_error": s2c_mean_pct,
        "s2c_tier1_mean_jaccard": s2c_mean_j,
        "s2c_tier1_mean_precision": s2c_mean_p,
        "s2c_tier1_mean_recall": s2c_mean_r,
        
        "c2s_rmse_size": c2s_rmse,
        "c2s_mae_size": c2s_mae,
        "c2s_mean_percentage_error": c2s_mean_pct,
        "c2s_tier1_mean_jaccard": c2s_mean_j,
        "c2s_tier1_mean_precision": c2s_mean_p,
        "c2s_tier1_mean_recall": c2s_mean_r,
        
        "avg_rmse_size": avg_rmse,
        "avg_mae_size": avg_mae,
        "avg_mean_percentage_error": avg_mean_pct,
        "avg_tier1_mean_jaccard": avg_mean_j,
        "avg_tier1_mean_precision": avg_mean_p,
        "avg_tier1_mean_recall": avg_mean_r,
    }
    return aggregate_metrics


def evaluate_pipeline_results(disease: str, synth_json: str = "synthetic_patients.json") -> dict:
    synth_path = Path(synth_json)
    cluster_path = CLUSTER_JSON
    
    if not synth_path.exists():
        raise FileNotFoundError(f"Required synthetic data file '{synth_json}' not found.")
        
    with open(synth_path, 'r', encoding='utf-8') as f:
        synth_data = json.load(f)
        
    with open(cluster_path, 'r', encoding='utf-8') as f:
        cluster_data = json.load(f)
        
    eval_metadata = synth_data.get("evaluation_metadata", {})
    gene_names = [disease]
    
    subgroups = {}
    target_genes = gene_names if gene_names else eval_metadata.keys()
    
    for g in target_genes:
        if g in eval_metadata:
            g_data = eval_metadata[g]
            if "subgroups" in g_data:
                for sg_id, sg_data in g_data["subgroups"].items():
                    composite_key = f"{g} (Subgroup {sg_id})"
                    subgroups[composite_key] = {
                        "parent_disease": g,
                        "actual_size": sg_data.get("actual_size", 0),
                        "signature_symptoms": sg_data.get("associated_symptoms", [])
                    }
                
    if not subgroups and eval_metadata:
        for g_key, g_val in eval_metadata.items():
            if isinstance(g_val, dict):
                if "subgroups" in g_val:
                    for sg_id, sg_data in g_val["subgroups"].items():
                        composite_key = f"{g_key} (Subgroup {sg_id})"
                        subgroups[composite_key] = {
                            "parent_disease": g_key,
                            "actual_size": sg_data.get("actual_size", 0),
                            "signature_symptoms": sg_data.get("associated_symptoms", [])
                        }

    optimal_config = cluster_data.get("clustering_results", {}).get("optimal", {})
    # Extract all clusters and filter out the noise cluster (-1)
    all_incnoise_clusters = optimal_config.get("clusters", [])
    detected_clusters = [c for c in all_incnoise_clusters if c.get("cluster_id") != -1]

    k = optimal_config.get("num_clusters_k", len(detected_clusters))

    if not subgroups or not detected_clusters:
        print("[!] Warning: Missing subgroups or optimal clusters in JSON files.")
        return {}

    # Size errors are computed on assigned patients only (noise does not count).
    coverage, size_ref_exact = assign_reference_sizes(subgroups, synth_data, detected_clusters)
    print(f"[*] Coverage (patients assigned to a cluster): {coverage:.1%} | "
          f"reference sizes: {'exact (cluster membership)' if size_ref_exact else 'proportional to coverage'}")

    print(f"\n[*] Loaded {len(subgroups)} total generated subgroups across {k} detected optimal clusters.")
    
    # --- Phase 1: Population and Matching Setup ---
    if k == 1:
        sigs = [populate_sigs_from_prevalent_symptoms(dc) for dc in detected_clusters]
        optimal_matching = create_trivial_single_cluster_matching(subgroups, detected_clusters, eval_metadata)
    else:
        sigs = compute_cluster_signatures(detected_clusters, k)
        optimal_matching = find_optimal_cluster_matching(sigs, subgroups, detected_clusters, eval_metadata)

    # --- Phase 2: Unified Metric Calculation Block ---
    metrics = calculate_subgroup_metrics(
        signatures=sigs, 
        matching=optimal_matching, 
        subgroups=subgroups,
        detected_clusters=detected_clusters,
        eval_metadata=eval_metadata,
        k=k
    )
    metrics["coverage"] = coverage
    metrics["size_reference_exact"] = float(size_ref_exact)

    return metrics


def passes_all(runs: list[dict]) -> bool:
    for key, (thr, kind) in THRESHOLDS.items():
        m = float(np.mean([r[key] for r in runs]))
        if not np.isfinite(m) or (kind == "min" and m < thr) or (kind == "max" and m > thr):
            return False
    return True


def is_close_call(runs: list[dict]) -> bool:
    """True if any metric's mean lies within CLOSE_CALL_SE standard errors of its threshold."""
    n = len(runs)
    if n < 2:
        return True
    for key, (thr, _) in THRESHOLDS.items():
        vals = np.array([r[key] for r in runs], dtype=float)
        se = vals.std(ddof=1) / math.sqrt(n)
        if abs(vals.mean() - thr) < CLOSE_CALL_SE * se:
            return True
    return False


def _record(result):
    """Counts consecutive failed iterations and aborts the grid if the setup is clearly broken."""
    global _consecutive_failures
    _consecutive_failures = 0 if result else _consecutive_failures + 1
    if _consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
        print(f"\n[!] HALT: {MAX_CONSECUTIVE_FAILURES} iterations failed in a row. "
              "Fix the error shown above, then restart with --resume.")
        sys.exit(1)
    return result


def print_iteration_metrics(m: dict):
    """One iteration: directional averages plus both directions, and coverage."""
    print(f"[+] Metrics: Jaccard {m['avg_tier1_mean_jaccard']:.3f} | precision {m['avg_tier1_mean_precision']:.3f} | "
          f"recall {m['avg_tier1_mean_recall']:.3f} | size error {m['avg_mean_percentage_error']:.1f}% | "
          f"RMSE {m['avg_rmse_size']:.2f} | MAE {m['avg_mae_size']:.2f} | coverage {m['coverage']:.1%}")
    print(f"    s2c: J {m['s2c_tier1_mean_jaccard']:.3f}  P {m['s2c_tier1_mean_precision']:.3f}  "
          f"R {m['s2c_tier1_mean_recall']:.3f}  size err {m['s2c_mean_percentage_error']:.1f}%   "
          f"c2s: J {m['c2s_tier1_mean_jaccard']:.3f}  P {m['c2s_tier1_mean_precision']:.3f}  "
          f"R {m['c2s_tier1_mean_recall']:.3f}  size err {m['c2s_mean_percentage_error']:.1f}%")


def print_cell_summary(runs: list[dict]):
    """Mean ± SD of each pass-criterion metric across iterations, with its threshold and verdict."""
    names = {"avg_tier1_mean_jaccard": "Jaccard", "avg_tier1_mean_precision": "Precision",
             "avg_tier1_mean_recall": "Recall", "avg_mean_percentage_error": "Size error (%)"}
    for key, (thr, kind) in THRESHOLDS.items():
        vals = np.array([r[key] for r in runs], dtype=float)
        mean = float(np.nanmean(vals))
        sd = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        ok = np.isfinite(mean) and (mean >= thr if kind == "min" else mean <= thr)
        sign = ">=" if kind == "min" else "<="
        print(f"    {names[key]:<15} {mean:8.3f} ± {sd:.3f}   (needs {sign} {thr:g})  {'OK' if ok else 'FAIL'}")
    cov = np.array([r.get("coverage", np.nan) for r in runs], dtype=float)
    print(f"    {'Coverage':<15} {np.nanmean(cov):8.1%}   (information only)")


def _load_paird():
    """Imports pairD_phenome.py once, from the location 'which pairD_phenome.py' points to."""
    global _PAIRD_MODULE
    if _PAIRD_MODULE is None:
        exe = shutil.which("pairD_phenome.py")
        if not exe:
            raise FileNotFoundError("pairD_phenome.py not found on PATH.")
        real = os.path.realpath(exe)                    # follow the ~/.local/bin symlink
        script_dir = os.path.dirname(real)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)              # so 'import pgmodules' works
        spec = importlib.util.spec_from_file_location("pairD_phenome", real)
        module = importlib.util.module_from_spec(spec)
        sys.modules["pairD_phenome"] = module           # needed by its worker processes
        spec.loader.exec_module(module)
        print(f"[*] Loaded pairD_phenome.py in-process from {real}")
        # Make sure the distance setting requested for the grid is the one in effect
        pgm = sys.modules.get("pgmodules")
        wanted = os.environ.get("PGM_ORGAN_MODE", "nearest")
        actual = getattr(pgm, "ORGAN_MODE", None)
        if actual != wanted:
            raise RuntimeError(
                f"pgmodules.py in use ({getattr(pgm, '__file__', '?')}) has organ mode {actual!r}, "
                f"but the grid requested {wanted!r}. Install the updated pgmodules.py next to pairD_phenome.py.")
        print(f"[*] dPhenome organ-system handling: {actual} ({pgm.__file__})")
        _PAIRD_MODULE = module
    return _PAIRD_MODULE


def _cluster_in_process(paird_args) -> bool:
    """
    Calls pairD_phenome.main() with the given arguments. pairD_phenome.py redirects
    sys.stdout/sys.stderr to a log file and never restores or closes it, so both are
    restored and the log closed afterwards (otherwise open files pile up over the grid).
    """
    module = _load_paird()
    saved_argv, saved_out, saved_err = sys.argv, sys.stdout, sys.stderr
    sys.argv = ["pairD_phenome.py"] + list(paird_args)
    ok = True
    try:
        module.main()
    except SystemExit as e:
        ok = e.code in (None, 0)
        if not ok:
            print(f"[!] pairD_phenome.py exited with code {e.code}", file=saved_out)
    except Exception as e:
        ok = False
        print(f"[!] EXCEPTION in pairD_phenome.py: {e}", file=saved_out)
        traceback.print_exc(file=saved_out)
    finally:
        for stream in (sys.stdout, sys.stderr):
            log = getattr(stream, "log", None)
            if log is not None and not log.closed:
                try:
                    log.close()
                except Exception:
                    pass
        sys.argv, sys.stdout, sys.stderr = saved_argv, saved_out, saved_err
    return ok


def _paird_result_dirs() -> set:
    """pairD_phenome.py's per-run folders: <yymmdd>_<genes>_<system>_<pairs> containing its outputs."""
    found = set()
    for p in Path('.').iterdir():
        if p.is_dir() and p.name[:6].isdigit() and p.name[6:7] == "_" and \
                ((p / "cdm.csv").exists() or (p / "pipeline.log").exists()):
            found.add(p.resolve())
    return found


def _remove_new_result_dirs(before: set):
    if KEEP_RESULTS:
        return
    for d in _paird_result_dirs() - before:
        shutil.rmtree(d, ignore_errors=True)


# =============================================================================
# TEST 1 grid
# =============================================================================

DISEASES = ["D1S0", "D2S2", "D3S3"]
N_PATIENTS = [30, 90, 180, 360]
SIG_SIZES = [3, 6, 9]
CONTRASTS = {"strong": (0.80, 0.05), "weak": (0.60, 0.15)}
REPORTING = [1.0]
LAYOUTS = ["dispersed", "concentrated"]
SUBSTITUTION = [0.0, 0.25, 0.5]
DISTANCES = ["dphenome", "jaccard", "resnik"]          # "hamming" also available
DISTANCE_CHOICES = ["dphenome", "jaccard", "resnik", "hamming"]

EXTRA_METRICS = ["n_clusters", "k_correct", "ari"]
METRICS = BASE_METRICS + EXTRA_METRICS
STATS = STAT_SUFFIXES
AXES = ["composition", "disease", "n_patients", "sig_size", "contrast", "reporting", "layout",
        "substitution", "distance"]
CSV_FIELDS = (AXES + ["n_true_subgroups", "high", "low", "status", "pass_all", "close_call",
                      "iterations_run", "successful_iterations"]
              + [f"{m}_{s}" for m in METRICS for s in STATS])
RAW_FIELDS = AXES + ["iteration", "seed"] + METRICS

SYNTH_JSON = "synthetic_patients.json"

# current grid point, read by the generation step
_CURRENT = {}


def cell_seed(base, disease, n, k, contrast, r, iteration):
    """Generation seed. Deliberately independent of layout, substitution and distance, so the
    same underlying patients are used across those axes (paired comparisons)."""
    return (zlib.crc32(f"{disease}|{n}|{k}|{contrast}|{r:g}|{iteration}".encode()) ^ base) & 0x7FFFFFFF


def generate_cohort() -> bool:
    c = _CURRENT
    try:
        gt.generate(c["table"], c["disease"], c["n"], c["k"], c["high"], c["low"],
                    null_like=c["null_like"], seed=c["seed"], output_json=SYNTH_JSON, verbose=True,
                    reporting=c["r"], layout=c["layout"], substitution=c["subst"], hpo_terms=c["hpo_terms"])
        return True
    except Exception as e:
        print(f"[!] Error in TEST 1 patient generation: {e}")
        traceback.print_exc()
        return False


def set_distance(mode):
    """Distance used by pgmodules.dPhenome() in the next pairD_phenome.py run."""
    _load_paird()
    pgm = sys.modules["pgmodules"]
    if not hasattr(pgm, "DISTANCE_MODE") or mode not in getattr(pgm, "DISTANCE_MODES", ()):
        raise RuntimeError(f"pgmodules.py in use ({pgm.__file__}) does not support distance {mode!r}: "
                           "install the updated pgmodules.py next to pairD_phenome.py.")
    if mode == "resnik":
        desc = pgm.resnik_ic_description()      # raises if phenotype.hpoa is missing
        if not _RESNIK_REPORTED:
            print(f"[*] Resnik distance with {desc}")
            _RESNIK_REPORTED.append(True)
    pgm.DISTANCE_MODE = mode          # read at call time; pairD's worker processes are forked after this


_RESNIK_REPORTED = []


def adjusted_rand(true, pred):
    """Adjusted Rand index (Hubert & Arabie). 1 = identical partitions, ~0 = chance."""
    from math import comb
    true, pred = list(true), list(pred)
    n = len(true)
    if n < 2:
        return float("nan")
    cont = {}
    for t, p in zip(true, pred):
        cont[(t, p)] = cont.get((t, p), 0) + 1
    a, b = {}, {}
    for (t, p), v in cont.items():
        a[t] = a.get(t, 0) + v
        b[p] = b.get(p, 0) + v
    sum_ij = sum(comb(v, 2) for v in cont.values())
    sum_a = sum(comb(v, 2) for v in a.values())
    sum_b = sum(comb(v, 2) for v in b.values())
    expected = sum_a * sum_b / comb(n, 2)
    max_index = (sum_a + sum_b) / 2
    if max_index == expected:          # both partitions trivial (e.g. one class, one cluster)
        return 1.0
    return (sum_ij - expected) / (max_index - expected)


def partition_metrics(n_true):
    """n_clusters, k_correct and ARI from the current cluster JSON and patient file."""
    with open(CLUSTER_JSON, encoding="utf-8") as f:
        clusters = json.load(f)["clustering_results"]["optimal"]["clusters"]
    with open(SYNTH_JSON, encoding="utf-8") as f:
        truth = {str(p["id"]): p["subgroup"] for p in json.load(f)["patients"]}
    real = [c for c in clusters if c.get("cluster_id") != -1]
    t, p = [], []
    for c in real:
        for pid in (_cluster_members(c) or []):
            if pid in truth:
                t.append(truth[pid])
                p.append(c["cluster_id"])
    return {"n_clusters": float(len(real)), "k_correct": float(len(real) == n_true),
            "ari": adjusted_rand(t, p) if t else float("nan")}


def run_iteration(paird_args, disease, n_true, distance):
    """Generate a cohort, cluster it with pairD_phenome.py, score the optimal clustering."""
    before = _paird_result_dirs()
    try:
        if CLUSTER_JSON.exists():
            CLUSTER_JSON.unlink()                       # never evaluate a stale result
        if not generate_cohort():
            print("[!] Warning: patient generation failed. Skipping iteration.")
            return _record(None)
        set_distance(distance)
        if not _cluster_in_process(paird_args) or not CLUSTER_JSON.exists():
            print("[!] Warning: pairD_phenome clustering failed. Skipping iteration.")
            return _record(None)
        try:
            m = evaluate_pipeline_results(disease, SYNTH_JSON)
        except Exception as e:
            print(f"[!] EXCEPTION while evaluating pipeline results: {e}")
            traceback.print_exc()
            return _record(None)
        if not m:
            return _record(None)
        try:
            m.update(partition_metrics(n_true))
        except Exception as e:
            print(f"[!] Could not compute partition metrics: {e}")
            m.update({x: float("nan") for x in EXTRA_METRICS})
        return _record(m)
    finally:
        _remove_new_result_dirs(before)             # scored (or failed): folder no longer needed


def is_close_call_partition(runs, threshold=0.8):
    """Close call on the rate of runs recovering the true number of subgroups."""
    v = np.array([x["k_correct"] for x in runs], dtype=float)
    if len(v) < 2:
        return True
    se = v.std(ddof=1) / math.sqrt(len(v))
    return abs(v.mean() - threshold) < CLOSE_CALL_SE * se


def print_iteration(m):
    print_iteration_metrics(m)
    print(f"    clusters found {int(m['n_clusters'])} (correct: {bool(m['k_correct'])}) | ARI {m['ari']:.3f}")


def summarise(runs):
    row = {}
    for m in METRICS:
        v = np.array([x.get(m, np.nan) for x in runs], dtype=float)
        ok = np.isfinite(v)
        row[f"{m}_mean"] = float(np.nanmean(v)) if ok.any() else float("nan")
        row[f"{m}_sd"] = float(np.nanstd(v, ddof=1)) if ok.sum() > 1 else 0.0
        row[f"{m}_median"] = float(np.nanmedian(v)) if ok.any() else float("nan")
        row[f"{m}_min"] = float(np.nanmin(v)) if ok.any() else float("nan")
        row[f"{m}_max"] = float(np.nanmax(v)) if ok.any() else float("nan")
    return row


def point_key(p):
    return (p["composition"], p["disease"], int(p["n_patients"]), int(p["sig_size"]), p["contrast"],
            float(p["reporting"]), p["layout"], float(p["substitution"]), p["distance"])


def load_done(path):
    done = set()
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["status"] != "failed":
                    done.add(point_key(row))
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-t", "--table", default="test1_table.csv")
    ap.add_argument("--composition", default=None,
                    help="Label written in the output (default: from the table name, e.g. C1).")
    ap.add_argument("-o", "--output", default="test1_grid_results.csv")
    ap.add_argument("--raw", default="test1_grid_iterations.csv", help="Per-iteration results.")
    ap.add_argument("--disease", nargs="+", default=DISEASES, choices=DISEASES)
    ap.add_argument("--n", nargs="+", type=int, default=N_PATIENTS)
    ap.add_argument("--k", nargs="+", type=int, default=SIG_SIZES)
    ap.add_argument("--contrast", nargs="+", default=["strong"], choices=list(CONTRASTS))
    ap.add_argument("--reporting", nargs="+", type=float, default=REPORTING,
                    help="Reporting rates r of the signature symptoms.")
    ap.add_argument("--layout", nargs="+", default=LAYOUTS, choices=LAYOUTS)
    ap.add_argument("--substitution", nargs="+", type=float, default=SUBSTITUTION,
                    help="Term-substitution probabilities p.")
    ap.add_argument("--distance", nargs="+", default=DISTANCES, choices=DISTANCE_CHOICES)
    ap.add_argument("--hpo-terms", default="hpo_terms.json", help="Needed when a substitution p > 0.")
    ap.add_argument("--null-like", choices=["D2S2", "D3S3"], default="D3S3",
                    help="D1S0: signature symptoms and flat frequency matched to this disease.")
    ap.add_argument("--base-seed", type=int, default=1)
    ap.add_argument("--iter-initial", type=int, default=N_ITER_INITIAL)
    ap.add_argument("--iter-max", type=int, default=N_ITER_MAX)
    ap.add_argument("--organ-mode", choices=["nearest", "penalty"], default="nearest")
    ap.add_argument("--resume", action="store_true", help="Skip grid points already in the output CSV.")
    ap.add_argument("--keep-results", action="store_true", help="Keep pairD_phenome.py's result folders.")
    ap.add_argument("--paird-args", default="",
                    help='Extra options passed to pairD_phenome.py, e.g. "--gridsearch 3 --min-dist 0". '
                         'Use a separate folder (or -o/--raw names) per setting: they are not part of the --resume key.')
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    comp = args.composition or Path(args.table).stem.split("_")[-1]
    if any(p > 0 for p in args.substitution) and not Path(args.hpo_terms).exists():
        sys.exit(f"{args.hpo_terms} not found (needed for term substitution).")
    k_max = {}
    for lay in args.layout:
        core, sig, k_max[lay] = gt.load_table(args.table, lay)
        too_big = [k for k in args.k if k > k_max[lay]]
        if too_big:
            sys.exit(f"k = {too_big} larger than the {k_max[lay]} {lay} signature ranks in {args.table}.")

    # D1S0 has no subgroups, so a contrast only matters through its flat frequency; contrasts
    # giving the same flat frequency (0.8/0.05 and 0.6/0.15 both give 0.30) would repeat the
    # same simulation, so only the first of them is run for D1S0.
    m = gt.DESIGNS[args.null_like]
    flat = {c: round((CONTRASTS[c][0] + (m - 1) * CONTRASTS[c][1]) / m, 6) for c in args.contrast}
    d1_contrasts = [c for i, c in enumerate(args.contrast) if flat[c] not in [flat[x] for x in args.contrast[:i]]]
    points = [dict(composition=comp, disease=d, n_patients=n, sig_size=k, contrast=c, reporting=r,
                   layout=lay, substitution=p, distance=dist)
              for d in args.disease for k in args.k
              for c in (d1_contrasts if d == "D1S0" else args.contrast)
              for r in args.reporting for lay in args.layout for p in args.substitution
              for n in args.n for dist in args.distance]
    if args.paird_args:
        print(f"pairD_phenome.py extra options: {args.paird_args}")
    print(f"=== TEST 1 grid, composition {comp}: {len(points)} grid points "
          f"({len(args.disease)} diseases x {len(args.n)} N x {len(args.k)} k x {len(args.contrast)} contrasts "
          f"x {len(args.reporting)} reporting rates x {len(args.layout)} layouts x {len(args.substitution)} "
          f"substitution levels x {len(args.distance)} distances), "
          f"{args.iter_initial}-{args.iter_max} iterations each ===")
    print(f"Table {args.table}: {len(core)} core symptoms; signature ranks " +
          ", ".join(f"{lay} 1..{k_max[lay]}" for lay in args.layout))
    for c in args.contrast:
        h, l = CONTRASTS[c]
        print(f"  {c}: high {h}, low {l}; D1S0 flat frequency {flat[c]:.3f}")
    if "D1S0" in args.disease and len(d1_contrasts) < len(args.contrast):
        print(f"  D1S0 run with contrast(s) {d1_contrasts} only: the others give the same flat frequency.")
    if args.dry_run:
        return

    # same set-up as run_gridsearch.py
    os.environ["PGM_ORGAN_MODE"] = args.organ_mode
    try:
        mp.set_start_method("fork", force=True)
    except (RuntimeError, ValueError) as e:
        print(f"[!] Could not set multiprocessing start method to 'fork': {e}")
    global KEEP_RESULTS
    KEEP_RESULTS = args.keep_results
    for dist in args.distance:                # loads pairD_phenome.py and checks pgmodules.py now
        set_distance(dist)
    out, raw = Path(args.output), Path(args.raw)
    done = load_done(out) if args.resume else set()
    append = args.resume and out.exists()

    t0 = time.perf_counter()
    with open(out, "a" if append else "w", newline="", encoding="utf-8") as f_out, \
            open(raw, "a" if (args.resume and raw.exists()) else "w", newline="", encoding="utf-8") as f_raw:
        w_out = csv.DictWriter(f_out, fieldnames=CSV_FIELDS)
        w_raw = csv.DictWriter(f_raw, fieldnames=RAW_FIELDS, extrasaction="ignore")
        if not append:
            w_out.writeheader()
        if not (args.resume and raw.stat().st_size > 0):
            w_raw.writeheader()

        for idx, pt in enumerate(points, 1):
            label = (f"{comp} {pt['disease']} N={pt['n_patients']} k={pt['sig_size']} {pt['contrast']} "
                     f"r={pt['reporting']:g} {pt['layout']} p={pt['substitution']:g} {pt['distance']}")
            if point_key(pt) in done:
                print(f"[=] {label}: already in {out}")
                continue
            disease, n, k, contrast, r = (pt["disease"], pt["n_patients"], pt["sig_size"], pt["contrast"],
                                          pt["reporting"])
            high, low = CONTRASTS[contrast]
            n_true = gt.DESIGNS[disease]
            print("\n" + "=" * 80)
            print(f"[{idx}/{len(points)}] {label}")
            print("=" * 80)

            paird_args = ["-f", SYNTH_JSON, "--gene", disease, "--skip-evaluation",
                          "--output-json", str(CLUSTER_JSON)] + shlex.split(args.paird_args)

            runs, attempts, close = [], 0, False

            def one():
                nonlocal attempts
                attempts += 1
                seed = cell_seed(args.base_seed, disease, n, k, contrast, r, attempts)
                _CURRENT.update(table=args.table, disease=disease, n=n, k=k, high=high, low=low,
                                null_like=args.null_like, seed=seed, r=r, layout=pt["layout"],
                                subst=pt["substitution"], hpo_terms=args.hpo_terms)
                print(f"--- Iteration {attempts} (seed {seed}) ---")
                res = run_iteration(paird_args, disease, n_true, pt["distance"])
                if res:
                    runs.append(res)
                    print_iteration(res)
                    w_raw.writerow({**pt, "iteration": attempts, "seed": seed, **res})
                    f_raw.flush()

            while attempts < args.iter_initial:
                one()
            # With term substitution the symptom metrics compare exact codes and drop by
            # construction, so extra iterations are decided on the partition instead.
            close_rule = is_close_call if pt["substitution"] == 0 else is_close_call_partition
            if len(runs) >= 2 and close_rule(runs):
                close = True
                print(f"[*] Close call after {len(runs)} runs: adding iterations up to {args.iter_max}.")
                while attempts < args.iter_max:
                    one()

            row = {**pt, "n_true_subgroups": n_true, "high": high, "low": low, "iterations_run": attempts,
                   "successful_iterations": len(runs), "close_call": close}
            if len(runs) < MIN_SUCCESSFUL:
                row.update(status="failed", pass_all=False)
            else:
                row.update(status="ok", pass_all=passes_all(runs), **summarise(runs))
            w_out.writerow(row)
            f_out.flush()
            print(f"[Summary] {label}: {len(runs)}/{attempts} successful, close_call={close}, "
                  f"pass={row['pass_all']}")
            if runs:
                print_cell_summary(runs)
                kc = np.mean([x["k_correct"] for x in runs])
                ari = np.nanmean([x["ari"] for x in runs])
                print(f"    {'Correct k':<15} {kc:8.0%}   | ARI {ari:.3f}   (information only)")

    print(f"\n[Complete] TEST 1 grid finished in {time.perf_counter() - t0:.0f} s. "
          f"Results: {out} (per grid point), {raw} (per iteration).")


if __name__ == "__main__":
    main()
