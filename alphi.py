#!/usr/bin/env python3.14
"""
================================================================================
alphi.py - ALPHI: agnostic latent phenotype identification (cohort selection, distances, clustering)
================================================================================
Description:
    This script executes a complete end-to-end workflow for phenotypic data:

    1. Cohort Selection: Reads patient data and HPO tree definitions, applying
       gene checks, default/optional DOI exclusion criteria, optional multi-gene/
       uncertain genetics exclusions, and unique symptom count thresholds.
    2. Pairwise Distance Calculation: Generates combinations and computes 
       pairwise distances concurrently using Python's multiprocessing pool.
       Decorates output records with primary human-readable term descriptors.
    3. Matrix Transformation: Reconstructs a squareform condensed distance matrix.
    4. Topological Reduction (Dynamic UMAP): Embeds distance vectors into a 
       manifold adaptively scaled to cohort size (e.g., 2D/3D for small cohorts,
       10D for large cohorts) to preserve topological relationships while avoiding
       high-dimensional spatial sparsity.
    5. Density Clustering (HDBSCAN): Executes a dynamic parameter grid search in
       embedding space scaled to dataset size, evaluates cluster quality (DBCV/Silhouette),
       and exports top-ranked solutions. Values >= 10 are strictly quantized to
       multiples of 5 to maintain methodology transparency.
    6. Export & Coherence Analysis: Writes condensed distance tables, dynamic 
       UMAP + cluster coordinate files, and an execution log (pipeline.log)
       to date-stamped output directories, then evaluates within-cluster phenotypic 
       Jaccard coherence, intra-cluster phenotypic distance, mean (SD) symptom count, 
       distinct DOI article count, distinct article publication year distributions 
       (oldest to newest), top enriched symptoms (ranked by Prevalence and PMI), 
       archetypal phenotype profiles (shared by >= 50% of cluster via parent_branches 
       DAG rollup), and multi-gene crosstables formatted as knitr::kable tables.
    7. JSON Visualization Export: Exports full cluster coordinate payloads
       (cluster IDs, human-readable labels, patient counts, patient article-level
       traceability records, hex color palettes, and matched x/y coordinate arrays)
       directly to a formatted JSON file suitable for web dashboard plotting and Plotly.js integrations.

CLI Option Explanations:
    --excluded-doi
        DOI string to exclude specific literature cohorts 
        (Default: "https://doi.org/10.1136/adc.43.232.646"). Pass empty string or 
        None to disable.

    --exclude-uncertain-genetics
        Flag to exclude patients for whom the underlying genetic cause is uncertain
        (i.e., patient records with multiple genes listed in their "gene" field).

    --digestive
        Controls the anatomical scope of the phenome distance calculation.
        - When omitted (default): Expands the distance calculation across all organ
          systems present in the HPO tree.
        - When specified (`--digestive`): Restricts phenome distance (`dPhenome`)
          calculations strictly to the digestive system organ branch.

    --gridsearch {1, 2, 3}
        Controls which stage of the UMAP-HDBSCAN hyperparameter space to optimize.
        - Mode 1 (default): Fixes UMAP `min_dist` (--min-dist) and executes a grid search 
          over HDBSCAN hyperparameters (`min_cluster_size` and `min_samples` dynamically 
          bounded based on cohort size N, constrained so min_samples <= min_cluster_size).
        - Mode 2: Fixes HDBSCAN parameters (`min_cluster_size` and `min_samples` derived 
          dynamically) and executes a grid search over UMAP manifold tightness (`min_dist` 
          across 10 values from 0.01 to 0.60).
        - Mode 3 (261028): joint search over the UMAP neighbourhood size and the HDBSCAN
          parameters. UMAP `n_neighbors` takes the values in --n-neighbors (default
          10, 15, 30, keeping those <= N - 1; the mode-1 value if none fits), each crossed
          with the mode-1 HDBSCAN grid (`min_dist` fixed, --min-dist). `n_neighbors` sets the
          scale of the structure UMAP preserves and `min_samples` the scale of HDBSCAN's
          density estimate: a group of m patients can only emerge if both are <= ~m.
          Configurations from different UMAP embeddings are ranked in the original
          distance space (see --score-space).

    --min-dist MIN_DIST
        UMAP `min_dist` for modes 1 and 3 (261028). Default 0, as UMAP's documentation
        recommends for clustering (with n_neighbors around 30, the mode-1 value for
        N > 50). Results before 261028 used 0.05: pass --min-dist 0.05 to reproduce them.
        Mode 2 varies it and ignores this option.

    --flag-delta D, --flag-top T, --flag-max-features M
        Few-feature split flag (261029; always computed, report only). For every PAIR of
        clusters of each retained clustering, the symptoms (exact codes) whose prevalence
        differs by >= D (default 0.3) between the two are counted. A pair is flagged when
        at most M (default 2) symptoms differ that much and the largest difference is >= T
        (default 0.5): the two clusters are then separated by one or two isolated symptoms
        rather than by a phenotype. Pairwise, so it catches both a single disease split in
        two and one true subgroup cut in two. Such splits arise with OB in large, completely
        reported cohorts when a symptom is a patient's only sign in its organ system
        (TEST 2 Wilson controls; TEST 1 C2, N = 360, k = 3). Written to the log,
        few_feature_flags.csv and the cluster JSON (flag_few_features,
        n_discriminating_features, closest_cluster, top_discriminating).
        A flagged cluster deserves scrutiny before being interpreted as a phenotype.

    --stability B
        Resampling stability of the retained clusterings (261028; Default 0 = off).
        For the optimal solution and each exported alternative: B times, 80 % of the
        patients (--stability-fraction) are drawn without replacement and clustered
        again with the same UMAP and HDBSCAN parameters (UMAP seed varied); each cluster's
        recovery is the best Jaccard overlap between its members in the subsample and a
        cluster found there (Hennig's clusterboot). Reported per cluster: mean Jaccard and
        share of resamples in which it dissolves (Jaccard <= 0.5); mean >= 0.75 is usually
        read as stable, <= 0.5 as not supported. Report only: the selection is unchanged.
        Written to the log, to stability.csv and to the cluster JSON.

    --score-space {umap, original}
        Space in which the selection step scores each configuration (261028).
        - umap: DBCV computed in the UMAP embedding (as before; default for modes 1, 2).
        - original: silhouette computed on pairD's own distance matrix, assigned
          patients only (default for mode 3). The same space for every configuration,
          so scores from different embeddings are comparable. The rest of the
          selection (noise discount, coverage fallback, stability penalty, cut-offs)
          is unchanged.

    --output-json OUTPUT_JSON
        Path for exporting the cluster coordinate data in JSON format for web/Plotly
        visualizations. Defaults to 'cluster_visualization.json' saved inside the
        run's date-stamped output directory.

Usage Examples:
    # Run default (MYO5B, All organ systems, default excluded DOI applied, HDBSCAN dynamic grid search)
    python alphi.py

    # Specify custom output JSON path
    python alphi.py --output-json ./my_visualization_data.json

    # Override DOI exclusion to a different paper
    python alphi.py --excluded-doi "https://doi.org/10.3390/jcm10030481"

    # Exclude uncertain genetics and restrict analysis to digestive system organ branch
    python alphi.py --gene MYO5B UNC45A STX3 --exclude-uncertain-genetics --digestive
================================================================================
"""

import os
import sys
import shutil
import time
import math
import argparse
import json
from datetime import datetime
from itertools import combinations
from collections import Counter
from typing import Dict
from pathlib import Path

import orjson
import numpy as np
import pandas as pd
import multiprocessing as mp
from scipy.spatial.distance import squareform
from scipy.stats import fisher_exact
from umap import UMAP
import hdbscan
from sklearn.metrics import silhouette_score
from tabulate import tabulate
from tqdm import tqdm

from pgmodules import dPhenome, extract_symptom_codes
import pgmodules
from term_selection import select_terms, norm as norm_code
import latent_classes
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

# ==========================================
# HPO DESCRIPTOR LOOKUP FEATURE
# ==========================================
SCRIPT_DIR = Path(__file__).resolve().parent

def resolve_resource(name: str) -> str:
    """Use the file in the current folder if present, else the one next to this script."""
    p = Path(name)
    if p.exists() or p.is_absolute():
        return str(p)
    fallback = SCRIPT_DIR / name
    return str(fallback) if fallback.exists() else str(p)

MAPPING_JSON_PATH = resolve_resource("id_to_term.json")


# ==========================================
# SOURCE ARTICLES OF A PATIENT RECORD
# ==========================================
def patient_articles(patient_entry: dict) -> list:
    """
    Source articles of a patient as a list of {"doi", "publication_year", "patient"} dicts.
    PytheasDB exports keep them in an "articles" list; older patient files had top-level
    "doi" (str or list), "publication_year" and "patients" fields, which are used when
    "articles" is absent.
    """
    arts = [a for a in patient_entry.get("articles") or [] if isinstance(a, dict)]
    if arts:
        return arts
    dois = patient_entry.get("doi") or []
    dois = dois if isinstance(dois, list) else [dois]
    year, label = patient_entry.get("publication_year"), patient_entry.get("patients", "")
    if not dois and year is None:
        return []
    return [{"doi": d, "publication_year": year, "patient": label} for d in (dois or [""])]

def patient_dois(patient_entry: dict) -> list:
    """All non-empty DOIs of a patient, in the order of its articles."""
    return [str(a["doi"]).strip() for a in patient_articles(patient_entry) if a.get("doi") and str(a["doi"]).strip()]

def patient_first_doi(patient_entry: dict) -> str:
    """DOI of the patient's first listed article ('' if none)."""
    dois = patient_dois(patient_entry)
    return dois[0] if dois else ""

def patient_label(patient_entry: dict) -> str:
    """Label of the patient in its first listed article ('' if none)."""
    arts = patient_articles(patient_entry)
    return str(arts[0].get("patient") or "") if arts else ""

def patient_pub_year(patient_entry: dict):
    """Earliest known publication year among the patient's articles (None if none)."""
    years = []
    for a in patient_articles(patient_entry):
        try:
            years.append(int(a.get("publication_year")))
        except (TypeError, ValueError):
            pass
    return min(years) if years else None

def check_patient_fields(patient_list, selected_ids, excluded_doi=None):
    """
    Reports how many selected patients carry the fields read after clustering (source
    article DOI and year, survival status and age) and how many records the excluded DOI
    matches. A renamed or reformatted field would otherwise fail silently: empty DOI
    columns, no article counts, no Cox models, no survival analysis, no exclusion.
    """
    lookup = {p.get("id"): p for p in patient_list}
    sel = [lookup[i] for i in selected_ids if i in lookup]
    n = len(sel)
    counts = {
        "DOI": sum(bool(patient_first_doi(p)) for p in sel),
        "publication year": sum(patient_pub_year(p) is not None for p in sel),
        "survival status and age": sum(parse_survival_data(p)[0] is not None for p in sel),
    }
    print("Record fields: " + ", ".join(f"{k} for {v}/{n}" for k, v in counts.items()) + " selected patients.")
    missing = [field for field, v in counts.items() if n and v == 0]
    if missing:
        print(f"  [WARNING] No {' / '.join(missing)} found in any selected record: the analyses that use "
              "them will be empty or skipped. ALPHI expects an 'articles' list of {\"doi\", "
              "\"publication_year\", \"patient\"} dicts (or top-level 'doi' and 'publication_year') and "
              "'alivedead' / 'alivedeadage'. Check the patient file's format.")
    if excluded_doi:
        norm = normalize_doi(excluded_doi)
        n_match = sum(norm in [normalize_doi(d) for d in patient_dois(p)] for p in patient_list)
        print(f"  Excluded DOI matches {n_match} record(s) in the patient file.")
        if n_match == 0:
            print("  [WARNING] The excluded DOI matches no record, so nothing was excluded. "
                  "Check the DOI or the patient file's format.")


def parse_survival_data(patient_entry: dict):
    """
    Parses 'alivedead' status and 'alivedeadage' duration from patient records.
    Returns: (time_in_years, event_occurred_bool)
    """
    raw_status = str(patient_entry.get("alivedead", "")).strip().lower()
    raw_age = patient_entry.get("alivedeadage")

    # Determine event occurrence (1 = Deceased, 0 = Censored/Alive)
    if raw_status in ["dead", "deceased", "1", "true"]:
        event = 1
    elif raw_status in ["alive", "0", "false"]:
        event = 0
    else:
        # If status missing or unparseable, return None to exclude from survival calculation
        return None, None

    # Parse age/time
    try:
        duration = float(raw_age)
        if duration < 0 or math.isnan(duration):
            return None, None
    except (ValueError, TypeError):
        return None, None

    return duration, event

def evaluate_cluster_survival(df_transformed: pd.DataFrame, patient_lookup: dict, cluster_col: str):
    """
    Performs Kaplan-Meier estimation and Log-Rank tests across clusters for a given clustering result.
    """
    print(f"\n==================================================")
    print(f"SURVIVAL ANALYSIS (KAPLAN-MEIER & LOG-RANK) FOR: {cluster_col}")
    print(f"==================================================")

    # Filter out noise points (-1)
    valid_df = df_transformed[df_transformed[cluster_col] != -1].copy()

    survival_records = []
    for _, row in valid_df.iterrows():
        pid = row["ids"]
        patient_entry = patient_lookup.get(pid, {})
        duration, event = parse_survival_data(patient_entry)

        if duration is not None and event is not None:
            survival_records.append({
                "id": pid,
                "cluster": row[cluster_col],
                "duration": duration,
                "event": event
            })

    df_surv = pd.DataFrame(survival_records)

    if df_surv.empty or len(df_surv["cluster"].unique()) < 2:
        print("  [Note] Insufficient survival data across multiple clusters to perform log-rank testing.")
        return

    # Overall Multivariate Log-Rank Test across all clusters
    try:
        overall_test = multivariate_logrank_test(
            event_durations=df_surv["duration"],
            groups=df_surv["cluster"],
            event_observed=df_surv["event"]
        )
        
        pval = overall_test.p_value
        p_val_str = "< 0.001" if pval < 0.001 else f"= {pval:.2g}"

        print(f"\nOverall Log-Rank Test across Clusters:")
        print(f"  -> Test Statistic (Chi-squared): {overall_test.test_statistic:.4f}")
        print(f"  -> p-value: {p_val_str} {'(Statistically Significant)' if pval < 0.05 else ''}")
    except Exception as e:
        print(f"  [Warning] Could not compute overall log-rank test: {e}")

    # Per-Cluster Kaplan-Meier Estimates
    kmf = KaplanMeierFitter()
    cluster_summary = []

    for cid in sorted(df_surv["cluster"].unique()):
        sub_df = df_surv[df_surv["cluster"] == cid]
        
        kmf.fit(sub_df["duration"], sub_df["event"], label=f"Cluster {cid}")
        
        median_survival = kmf.median_survival_time_
        median_str = f"{median_survival:.2f} yrs" if not np.isinf(median_survival) and not np.isnan(median_survival) else "Not Reached"
        
        # Survival probabilities at fixed landmarks
        surv_5yr = kmf.predict(5.0)
        surv_10yr = kmf.predict(10.0)

        cluster_summary.append({
            "Cluster": f"Cluster {cid}",
            "N Total": len(sub_df),
            "Deaths (Events)": sub_df["event"].sum(),
            "Censored (Alive)": (sub_df["event"] == 0).sum(),
            "Median Survival": median_str,
            "5-Year Survival (%)": f"{surv_5yr * 100:.1f}%",
            "10-Year Survival (%)": f"{surv_10yr * 100:.1f}%"
        })

    print("\nKaplan-Meier Survival Summary Table:")
    print(tabulate(cluster_summary, headers="keys", tablefmt="pipe"))

def evaluate_cox_ph_model(df_transformed: pd.DataFrame, patient_lookup: dict, cluster_col: str):
    """
    Fits multivariable Cox Proportional Hazards models evaluating relative mortality risk 
    across clusters, adjusting for symptom burden, continuous publication year, and 
    a dichotomized era step-effect (Pre-2000 vs. Post-2000).
    """
    print(f"\n==================================================")
    print(f"MULTIVARIABLE COX PH MODELS FOR: {cluster_col}")
    print(f"==================================================")

    # Exclude noise points (-1)
    valid_df = df_transformed[df_transformed[cluster_col] != -1].copy()

    cox_records = []
    for _, row in valid_df.iterrows():
        pid = row["ids"]
        patient_entry = patient_lookup.get(pid, {})
        duration, event = parse_survival_data(patient_entry)

        # Parse publication year
        pub_year = patient_pub_year(patient_entry)
        pub_year_val = float(pub_year) if pub_year is not None else np.nan

        if duration is not None and event is not None and not np.isnan(pub_year_val):
            cox_records.append({
                "cluster": str(row[cluster_col]), # Categorical factor (dummy-encoded)
                "symptom_count": row.get("symptoms", len(patient_entry.get("symptoms", []))),
                "pub_year": pub_year_val,
                "era_post2000": 1.0 if pub_year_val >= 2000 else 0.0,  # Test A: Binary Era Step
                "duration": duration,
                "event": event
            })

    df_cox = pd.DataFrame(cox_records)

    # Validate requirements
    if df_cox.empty or len(df_cox["cluster"].unique()) < 2 or df_cox["event"].sum() == 0:
        print("  [Note] Insufficient survival events or complete covariate records to fit Cox models.")
        return

    # Convert 'cluster' into dummy variables (lowest numerical ID as reference)
    df_encoded = pd.get_dummies(df_cox, columns=["cluster"], drop_first=True, dtype=float)

    def run_and_print_cox(model_df, title_label):
        cph = CoxPHFitter()
        try:
            cph.fit(model_df, duration_col="duration", event_col="event")
            summary_df = cph.summary.reset_index()
            
            cox_rows = []
            for _, row in summary_df.iterrows():
                covariate = row["covariate"]
                hr = row["exp(coef)"]
                ci_lower = row["exp(coef) lower 95%"]
                ci_upper = row["exp(coef) upper 95%"]
                pval = row["p"]

                p_val_str = "< 0.001" if pval < 0.001 else f"= {pval:.2g}"
                
                if covariate.startswith("cluster_"):
                    cluster_id = covariate.replace("cluster_", "")
                    covariate_label = f"Cluster {cluster_id} (vs Ref)"
                elif covariate == "symptom_count":
                    covariate_label = "Symptom Count (Adj)"
                elif covariate == "pub_year":
                    covariate_label = "Publication Year (Continuous)"
                elif covariate == "era_post2000":
                    covariate_label = "Era (Post-2000 vs Pre-2000)"
                else:
                    covariate_label = covariate

                cox_rows.append({
                    "Variable / Factor": covariate_label,
                    "Hazard Ratio (HR)": f"{hr:.2f}",
                    "95% CI": f"[{ci_lower:.2f} - {ci_upper:.2f}]",
                    "p-value": p_val_str,
                    "Significance": "(Significant)" if pval < 0.05 else ""
                })

            print(f"\n{title_label} (N = {len(df_cox)} Patients, {int(df_cox['event'].sum())} Events):")
            print(tabulate(cox_rows, headers="keys", tablefmt="pipe"))
            print(f"  -> Model Concordance (C-index): {cph.concordance_index_:.3f}")
            print(f"  -> Log-Likelihood Ratio Test p-value: {cph.log_likelihood_ratio_test().p_value:.2g}")

        except Exception as e:
            print(f"  [Warning] Could not fit {title_label}: {e}")

    # 1. Primary Model (Continuous Year)
    cols_primary = [c for c in df_encoded.columns if c != "era_post2000"]
    run_and_print_cox(df_encoded[cols_primary], "Model 1: Continuous Publication Year")

    # 2. Test A Model (Binary Era Step-Effect)
    cols_era = [c for c in df_encoded.columns if c != "pub_year"]
    run_and_print_cox(df_encoded[cols_era], "Model 2 (Test A): Binary Era Step-Effect (Pre-2000 vs Post-2000)")

    # 3. Model A2 (Both Binary Era and Continuous Year)
    run_and_print_cox(df_encoded, "Model 3: Combined Era Step-Effect + Continuous Residual Trend")


def evaluate_cox_ph_model_continuous(df_transformed: pd.DataFrame, patient_lookup: dict, cluster_col: str):
    """
    Fits a multivariable Cox Proportional Hazards model evaluating relative mortality risk 
    across clusters, adjusting for total symptom burden and publication year.
    """
    print(f"\n==================================================")
    print(f"MULTIVARIABLE COX PH MODEL FOR: {cluster_col}")
    print(f"==================================================")

    # Exclude noise points (-1)
    valid_df = df_transformed[df_transformed[cluster_col] != -1].copy()

    cox_records = []
    for _, row in valid_df.iterrows():
        pid = row["ids"]
        patient_entry = patient_lookup.get(pid, {})
        duration, event = parse_survival_data(patient_entry)

        # Parse publication year
        pub_year = patient_pub_year(patient_entry)
        pub_year_val = float(pub_year) if pub_year is not None else np.nan

        if duration is not None and event is not None and not np.isnan(pub_year_val):
            cox_records.append({
                "cluster": str(row[cluster_col]), # Categorical factor (dummy-encoded)
                "symptom_count": row.get("symptoms", len(patient_entry.get("symptoms", []))),
                "pub_year": pub_year_val,
                "duration": duration,
                "event": event
            })

    df_cox = pd.DataFrame(cox_records)

    # Validate requirements (at least 2 distinct clusters, at least 1 observed event)
    if df_cox.empty or len(df_cox["cluster"].unique()) < 2 or df_cox["event"].sum() == 0:
        print("  [Note] Insufficient survival events or complete covariate records to fit multivariable Cox model.")
        return

    # Convert 'cluster' into dummy/one-hot categorical variables (lowest numerical ID as reference)
    df_encoded = pd.get_dummies(df_cox, columns=["cluster"], drop_first=True, dtype=float)

    cph = CoxPHFitter()
    try:
        # Fit multivariable model
        cph.fit(df_encoded, duration_col="duration", event_col="event")
        
        summary_df = cph.summary.reset_index()
        
        cox_rows = []
        for _, row in summary_df.iterrows():
            covariate = row["covariate"]
            hr = row["exp(coef)"]
            ci_lower = row["exp(coef) lower 95%"]
            ci_upper = row["exp(coef) upper 95%"]
            pval = row["p"]

            # Format p-value dynamically
            p_val_str = "< 0.001" if pval < 0.001 else f"= {pval:.2g}"
            
            # Format display labels
            if covariate.startswith("cluster_"):
                cluster_id = covariate.replace("cluster_", "")
                covariate_label = f"Cluster {cluster_id} (vs Ref)"
            elif covariate == "symptom_count":
                covariate_label = "Symptom Count (Adj)"
            elif covariate == "pub_year":
                covariate_label = "Publication Year (Adj)"
            else:
                covariate_label = covariate

            cox_rows.append({
                "Variable / Factor": covariate_label,
                "Hazard Ratio (HR)": f"{hr:.2f}",
                "95% CI": f"[{ci_lower:.2f} - {ci_upper:.2f}]",
                "p-value": p_val_str,
                "Significance": "(Significant)" if pval < 0.05 else ""
            })

        print(f"\nMultivariable Cox Proportional Hazards Regression (N = {len(df_cox)} Patients, {int(df_cox['event'].sum())} Events):")
        print(tabulate(cox_rows, headers="keys", tablefmt="pipe"))

    except Exception as e:
        print(f"  [Warning] Could not fit Cox PH model: {e}")



def load_hpo_descriptors(mapping_file: str) -> Dict[str, str]:
    """
    Loads the key-value dictionary mapping HPO IDs (e.g. 'HP:0000002') 
    to their primary string descriptors/labels.
    """
    if not os.path.exists(mapping_file):
        print(f"[Warning] Descriptor map '{mapping_file}' not found. "
              f"Descriptors will fallback to 'Unknown Term'.", file=sys.stderr)
        return {}
    
    with open(mapping_file, "r", encoding="utf-8") as f:
        return orjson.loads(f.read())


# Load descriptor mapping globally for instant O(1) lookups across process workers
ID_TO_TERM_MAP: Dict[str, str] = load_hpo_descriptors(MAPPING_JSON_PATH)



def get_hpo_label(hpo_id: str, tree_lookup: dict = None) -> str:
    """
    Look up the human-readable descriptor for a given HPO code using the JSON
    lookup map, falling back to tree_lookup or 'Unknown Term' if unmapped.
    """
    code_formatted = f"HP:{str(hpo_id).replace('HP:', '').zfill(7)}"
    
    if code_formatted in ID_TO_TERM_MAP:
        return ID_TO_TERM_MAP[code_formatted]
    
    if tree_lookup:
        code_raw = str(hpo_id).replace("HP:", "").zfill(7)
        tree_info = tree_lookup.get(code_raw, {})
        if tree_info:
            return tree_info.get("description", tree_info.get("name", "Unknown Term"))

    return "Unknown Term"


# ==========================================
# DUAL TEE LOGGER FOR CONSOLE & LOG FILE
# ==========================================
class TeeLogger:
    """Tee logger to write stdout/stderr to both console and a log file simultaneously."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ==========================================
# DBCV SCORE COMPUTATION WITH JITTER GUARD
# ==========================================
def principal_coordinates(D, n_components):
    """Classical multidimensional scaling (principal coordinates analysis) of a distance matrix:
    coordinates whose Euclidean distances best reproduce D, on the axes of largest variance.
    Deterministic; used for the output coordinates with --no-embedding."""
    D = np.asarray(D, dtype=np.float64)
    n = len(D)
    J = np.eye(n) - 1.0 / n
    B = -0.5 * J @ (D ** 2) @ J
    w, v = np.linalg.eigh(B)
    order = np.argsort(w)[::-1][:n_components]
    w, v = np.clip(w[order], 0, None), v[:, order]
    coords = v * np.sqrt(w)
    if coords.shape[1] < n_components:                      # fewer axes than requested
        coords = np.hstack([coords, np.zeros((n, n_components - coords.shape[1]))])
    return coords


def compute_dbcv(X_embed, labels):
    """
    Computes Density-Based Cluster Validation (DBCV) score using hdbscan.validity.
    Includes a noise/jitter guard to prevent division-by-zero errors when
    embeddings contain overlapping/identical low-dimensional coordinates.
    """
    unique_labels = set(labels)
    num_clusters = len(unique_labels - {-1})
    if num_clusters < 2:
        return np.nan

    try:
        np.random.seed(42)
        jittered_X = X_embed.astype(np.float64) + np.random.normal(0, 1e-7, X_embed.shape)
        
        dbcv_val = hdbscan.validity.validity_index(
            jittered_X, 
            labels.astype(np.int64), 
            metric="euclidean"
        )
        return float(dbcv_val) if not np.isnan(dbcv_val) else np.nan
    except Exception:
        return np.nan


# ==========================================
# DYNAMIC HYPERPARAMETER COMPUTATION
# ==========================================
def snap_grid_param(val: float) -> int:
    """
    Ensures parameters < 10 retain integer precision, while parameters >= 10 
    are quantized in steps of 5 to avoid over-tuning suspicions in peer review.
    """
    ival = max(1, int(round(val)))
    if ival < 10:
        return ival
    return int(round(ival / 5.0) * 5)

def compute_dynamic_parameters(n_samples):
    """
    Computes dataset-size-aware parameters for UMAP and HDBSCAN grid search.
    Enforces mcs [5..25] scaled by N/10, with ms bounded between max(5, mcs - 5) and mcs.
    """
    if n_samples < 2:
        raise ValueError(f"Cohort size N={n_samples} is too small for clustering.")

    # 1. Dynamic UMAP Dimensions & Neighbors
    if n_samples < 30:
        n_components = 2
    elif n_samples < 100:
        n_components = 3
    else:
        n_components = 10

    max_neighbors = max(2, n_samples - 1)
    if n_samples <= 15:
        n_neighbors = max(2, n_samples - 2)
    elif n_samples <= 50:
        n_neighbors = min(10, max_neighbors)
    else:
        n_neighbors = min(30, max_neighbors)

    # 2. HDBSCAN Grid Search Bounds with ms <= mcs enforcement
    max_mcs = int(max(5, min(25, round(n_samples / 10.0) * 5)))
    mcs_values = [v for v in range(5, 26, 5) if v <= max_mcs and v <= n_samples]
    if not mcs_values:
        mcs_values = [5]

    grid_pairs = []
    for mcs in mcs_values:
        min_ms = max(5, mcs - 5)
        max_ms = mcs  # Enforces that ms never exceeds mcs
        ms_options = [ms for ms in range(min_ms, max_ms + 1, 5) if ms <= n_samples]
        for ms in ms_options:
            grid_pairs.append((mcs, ms))

    unique_mcs = sorted(list(set(p[0] for p in grid_pairs)))
    unique_ms = sorted(list(set(p[1] for p in grid_pairs)))

    # Defaults
    raw_base_mcs = np.sqrt(n_samples)
    base_mcs = int(round(raw_base_mcs / 5.0) * 5)
    default_mcs = max(5, min(max_mcs, base_mcs))
    default_ms = max(5, default_mcs - 5)
    if default_ms > default_mcs:
        default_ms = default_mcs

    return {
        "n_components": n_components,
        "n_neighbors": n_neighbors,
        "max_neighbors": max_neighbors,
        "min_cluster_sizes": unique_mcs,
        "min_samples_list": unique_ms,
        "grid_pairs": grid_pairs,
        "default_mcs": default_mcs,
        "default_ms": default_ms
    }


# ==========================================
# COHORT ARCHETYPAL PHENOTYPE EXTRACTION UTILITY
# ==========================================
def extract_cohort_archetypal_phenotypes(
    df_transformed: pd.DataFrame,
    patient_lookup: dict,
    tree_lookup: dict,
    threshold: float = 0.50,
    min_depth_from_root: int = 3,  # Retained for signature compatibility
) -> list:
    """Extracts an archetypal phenotypic signature across the entire patient cohort

    strictly using direct (as-reported) symptoms shared by >= threshold of all
    cohort patients (no DAG ancestor rollups).
    """
    cohort_patient_ids = df_transformed["ids"].tolist()
    total_patients = len(cohort_patient_ids)

    if total_patients == 0:
        return []

    # --- STEP 1: Compute Direct (As-Reported) Counts across Cohort ---
    direct_patient_counts = Counter()

    for pid in cohort_patient_ids:
        p_entry = patient_lookup.get(pid, {})
        direct_symptoms = set(
            extract_symptom_codes(p_entry.get("symptoms", []))
        )
        clean_direct = {
            str(c).replace("HP:", "").zfill(7) for c in direct_symptoms
        }
        direct_patient_counts.update(clean_direct)

    # --- STEP 2: Filter by Threshold & Build Payload ---
    cohort_archetypes = []

    for node, count in direct_patient_counts.items():
        prev = count / total_patients
        if prev >= threshold:
            hpo_code_formatted = (
                f"HP:{node}" if not str(node).startswith("HP:") else node
            )

           
            cohort_archetypes.append(
                {
                    "HPO Code": hpo_code_formatted,
                    "Cohort Prevalence": f"{round(prev * 100)}%",
                    "_prev_raw": prev,
                }
            )

    # --- STEP 3: Sort by Prevalence Descending ---
    cohort_archetypes.sort(key=lambda x: x["_prev_raw"], reverse=True)

    for item in cohort_archetypes:
        del item["_prev_raw"]

    return cohort_archetypes

# ==========================================
# ARCHETYPAL PHENOTYPE EXTRACTION UTILITY
# ==========================================
def extract_archetypal_phenotypes(
    target_cluster_id,
    df_transformed,
    cluster_col,
    patient_lookup,
    tree_lookup,
    threshold=0.50,
    min_depth_from_root=3,
):
    """Extracts an archetypal symptom profile for a specific target cluster by traversing parent_branches.

    - Preserves child terms if directly reported above threshold.
    - Blocks ancestor rollups above terms that already independently meet the threshold.
    """
    valid_df = df_transformed[df_transformed[cluster_col] != -1]

    unique_clusters = sorted(valid_df[cluster_col].unique())
    if target_cluster_id not in unique_clusters or valid_df.empty:
        return []

    global_patient_ids = df_transformed["ids"].tolist()
    total_cohort_patients = len(global_patient_ids)

    cluster_patient_map = {
        cid: valid_df[valid_df[cluster_col] == cid]["ids"].tolist()
        for cid in unique_clusters
    }

    target_cluster_ids = cluster_patient_map[target_cluster_id]
    num_target_patients = len(target_cluster_ids)

    if num_target_patients == 0 or total_cohort_patients == 0:
        return []

    # --- STEP 1: Compute Direct (As-Reported) Counts for Target Cluster ---
    target_direct_counts = Counter()
    patient_direct_symptoms = []

    for pid in target_cluster_ids:
        p_entry = patient_lookup.get(pid, {})
        direct_symptoms = set(
            extract_symptom_codes(p_entry.get("symptoms", []))
        )
        clean_direct = {
            str(c).replace("HP:", "").zfill(7) for c in direct_symptoms
        }
        patient_direct_symptoms.append(clean_direct)
        target_direct_counts.update(clean_direct)

    # Identify direct terms in the target cluster that cross the threshold independently
    qualifying_direct_nodes = {
        node
        for node, count in target_direct_counts.items()
        if (count / num_target_patients) >= threshold
    }

    # --- STEP 2: Ancestor Expansion Function with Pruning Guard ---
    def get_node_ancestors_with_depth(symptom_set, stop_nodes_set=None):
        """Traverses parent branches up to min_depth_from_root.

        If a term in symptom_set is in stop_nodes_set, its ancestors are NOT
        added.
        """
        if stop_nodes_set is None:
            stop_nodes_set = set()

        node_depths = {}

        for code in symptom_set:
            code_str = str(code).replace("HP:", "").zfill(7)

            # Place the node itself in the depth map
            tree_info = tree_lookup.get(code_str, {})
            branches = tree_info.get("parent_branches", [])

            if not branches:
                node_depths[code_str] = max(node_depths.get(code_str, 0), 99)
                continue

            # Process branches
            for branch in branches:
                valid_branch = [c for c in branch if c != "-------"]

                if "0000001" in valid_branch:
                    root_idx = valid_branch.index("0000001")
                else:
                    root_idx = len(valid_branch) - 1

                for idx, ancestor_code in enumerate(valid_branch):
                    depth_from_root = root_idx - idx

                    # RULE (ii) GUARD: If an ancestor higher in the branch is reached,
                    # but the starting direct symptom (or a child) already qualifies,
                    # DO NOT roll up to its parents.
                    if idx > 0 and code_str in stop_nodes_set:
                        break  # Stop traversing up this branch

                    if (
                        depth_from_root >= min_depth_from_root or idx == 0
                    ):
                        if (
                            ancestor_code not in node_depths
                            or depth_from_root > node_depths[ancestor_code]
                        ):
                            node_depths[ancestor_code] = depth_from_root

        return node_depths

    # --- STEP 3: Traversal & Accumulation Across Clusters ---
    cluster_ancestor_counts = {cid: Counter() for cid in unique_clusters}
    global_ancestor_counts = Counter()

    for cid, p_ids in cluster_patient_map.items():
        for pid in p_ids:
            p_entry = patient_lookup.get(pid, {})
            d_symptoms = set(
                extract_symptom_codes(p_entry.get("symptoms", []))
            )

            # Stop parent rollups for symptoms in qualifying_direct_nodes
            anc_map = get_node_ancestors_with_depth(
                d_symptoms, stop_nodes_set=qualifying_direct_nodes
            )

            cluster_ancestor_counts[cid].update(anc_map.keys())
            global_ancestor_counts.update(anc_map.keys())

    # Calculate depths specifically for candidate extraction in target cluster
    patient_ancestor_maps = []
    for clean_direct in patient_direct_symptoms:
        anc_map = get_node_ancestors_with_depth(
            clean_direct, stop_nodes_set=qualifying_direct_nodes
        )
        patient_ancestor_maps.append(anc_map)

    node_patient_counts = cluster_ancestor_counts[target_cluster_id]
    node_max_depths = {}

    for p_map in patient_ancestor_maps:
        for node, depth in p_map.items():
            if (
                node not in node_max_depths
                or depth > node_max_depths[node]
            ):
                node_max_depths[node] = depth

    # --- STEP 4: Candidate Selection & DAG Redundancy Pruning ---
    raw_candidates = []
    for node, count in node_patient_counts.items():
        prev = count / num_target_patients
        depth = node_max_depths.get(node, 0)

        if prev >= threshold and (
            depth >= min_depth_from_root
            or target_direct_counts.get(node, 0) > 0
        ):
            raw_candidates.append(node)

    def get_all_ancestors(node_code):
        ancestors = set()
        clean_code = str(node_code).replace("HP:", "").zfill(7)
        tree_info = tree_lookup.get(clean_code, {})
        
        for branch in tree_info.get("parent_branches", []):
            valid_branch = [
                str(c).replace("HP:", "").zfill(7) 
                for c in branch 
                if c != "-------"
            ]
            # Add all nodes in the branch EXCEPT the candidate node itself
            ancestors.update({c for c in valid_branch if c != clean_code})
            
        return ancestors

    # Block any candidate node that is an ancestor of ANOTHER qualifying candidate node
    blocked_ancestors = set()
    for candidate in raw_candidates:
        blocked_ancestors.update(get_all_ancestors(candidate))

    # Retain only candidate nodes that are not subsumed by a qualifying descendant
    candidate_nodes = [
        node for node in raw_candidates if node not in blocked_ancestors
    ]

    candidate_nodes.sort(
        key=lambda x: (node_patient_counts[x], node_max_depths.get(x, 0)),
        reverse=True,
    )

    # --- STEP 5: Build Final Payload ---
    archetype_profile = []

    for node in candidate_nodes:
        direct_cnt = target_direct_counts.get(node, 0)
        total_cnt = node_patient_counts[node]

        direct_prev = direct_cnt / num_target_patients
        target_prev = total_cnt / num_target_patients

        p_cluster = target_prev
        g_count = global_ancestor_counts.get(node, 0)
        p_global = g_count / total_cohort_patients
        pmi = (
            math.log2(p_cluster / p_global)
            if p_global > 0 and p_cluster > 0
            else 0.0
        )

        if direct_prev >= threshold:
            rep_type = "As-Reported Term"
        else:
            rep_type = f"DAG Ancestor Rollup (Direct: {round(direct_prev * 100)}%)"

        hpo_code_formatted = (
            f"HP:{node}" if not str(node).startswith("HP:") else node
        )
        description = get_hpo_label(hpo_code_formatted, tree_lookup=tree_lookup)

        row_item = {
            "HPO Code": hpo_code_formatted,
            "Description": description,
        }

        for cid in unique_clusters:
            c_size = len(cluster_patient_map[cid])
            c_cnt = cluster_ancestor_counts[cid].get(node, 0)
            c_prev = (c_cnt / c_size) if c_size > 0 else 0.0
            row_item[f"Cluster {cid}"] = f"{round(c_prev * 100)}%"

        row_item["Archetype PMI"] = f"{pmi:.2f}"
        row_item["Representation Level"] = rep_type
        row_item["_pmi_raw"] = pmi
        row_item["_prev_raw"] = target_prev

        archetype_profile.append(row_item)

    archetype_profile.sort(
        key=lambda x: (x["_prev_raw"], x["_pmi_raw"]), reverse=True
    )

    for item in archetype_profile:
        del item["_pmi_raw"]
        del item["_prev_raw"]

    # --- ADAPTIVE CLUSTER-COUNT RELAXATION FILTERING ---
    max_off_cluster_threshold = 0.20  # Keep strict 20% max threshold
    num_total_clusters = len(unique_clusters)
    num_other_clusters = num_total_clusters - 1

    filtered_profile = []

    # allowed_violations:
    # 0 -> less than 0.2 in ALL other clusters (Valid for any cluster count)
    # 1 -> less than 0.2 in ALL BUT ONE other cluster (Requires > 1 total clusters)
    # 2 -> less than 0.2 in ALL BUT TWO other clusters (Requires > 2 total clusters), etc.
    for allowed_violations in range(num_other_clusters):
        # Enforce cluster count thresholds for relaxed violation levels
        if num_total_clusters <= (allowed_violations):
            continue

        filtered_profile = []

        for row in archetype_profile:
            # Calculate prevalence in all non-target clusters
            other_cluster_prevs = [
                float(row[f"Cluster {cid}"].replace("%", "")) / 100.0
                for cid in unique_clusters
                if cid != target_cluster_id
            ]

            # Count how many other clusters exceed the 20% threshold
            violations = sum(
                1
                for prev in other_cluster_prevs
                if prev >= max_off_cluster_threshold
            )

            # Keep term if the number of violating clusters is within allowed limit
            if violations <= allowed_violations:
                filtered_profile.append(row)

        # Stop as soon as we meet or exceed the target of 3 terms
        if len(filtered_profile) >= 3:
            break

    return filtered_profile



# ===========================================================
# SIGNIFICANTLY ASSOCIATED SYMPTOMS
# ===========================================================
def extract_associated_symptoms(
    target_cluster_id,
    target_direct_counts,
    global_symptom_counts,
    cluster_patient_map,
    total_cohort_patients,
    tree_lookup,
    max_prevalence=1.0,  # Considers all as-reported symptoms by default
    min_pmi=0.5,
    max_p_val=0.1,  # Returns all symptoms with p-value <= max_p_val
):
    """Extracts as-reported HPO symptoms associated with a target cluster,

    filtered by p-value threshold and ranked strictly by level of statistical association
    (Fisher's exact p-value ascending). No DAG ancestor rollup is performed.
    """
    num_target_patients = len(cluster_patient_map[target_cluster_id])
    num_other_patients = total_cohort_patients - num_target_patients

    if num_target_patients == 0 or num_other_patients == 0:
        return []

    candidate_stats = []

    # Iterate directly over as-reported symptom counts for the cluster
    for node, target_count in target_direct_counts.items():
        prev_target = target_count / num_target_patients

        # Filter by prevalence ceiling
        if 0 < prev_target <= max_prevalence:
            global_count = global_symptom_counts.get(node, 0)
            other_count = global_count - target_count

            p_global = global_count / total_cohort_patients
            pmi = (
                math.log2(prev_target / p_global)
                if p_global > 0 and prev_target > 0
                else 0.0
            )

            # Filter by PMI threshold
            if pmi >= min_pmi:
                # 2x2 Contingency Table for Fisher's Exact Test:
                # [[Target_With, Target_Without], [Other_With, Other_Without]]
                table = [
                    [target_count, num_target_patients - target_count],
                    [other_count, num_other_patients - other_count],
                ]

                # Calculate Fisher's Exact p-value (overrepresented in target cluster)
                _, p_val = fisher_exact(table, alternative="greater")

                # Filter by max_p_val threshold
                if p_val <= max_p_val:
                    candidate_stats.append(
                        {
                            "node": node,
                            "prev_target": prev_target,
                            "prev_other": (
                                other_count / num_other_patients
                                if num_other_patients > 0
                                else 0.0
                            ),
                            "pmi": pmi,
                            "p_val": p_val,
                            "target_count": target_count,
                            "other_count": other_count,
                        }
                    )

    if not candidate_stats:
        return []

    # Rank symptoms strictly by association (p-value ascending, then PMI, then prevalence)
    candidate_stats.sort(key=lambda x: (x["p_val"], -x["pmi"], -x["prev_target"]))

    payload = []

    # Helper function for p-value formatting
    def format_p_val(p):
        if p < 0.001:
            return "< 0.001"
        elif p < 0.01:
            return f"{p:.3f}"
        else:
            return f"{p:.2f}"

    for item in candidate_stats:
        node = item["node"]
        hpo_code = f"HP:{node}" if not str(node).startswith("HP:") else node

        payload.append(
            {
                "HPO Code": hpo_code,
                "Description": get_hpo_label(hpo_code, tree_lookup=tree_lookup),
                "Cluster Prev (%)": f"{round(item['prev_target'] * 100)}%",
                "Rest Cohort Prev (%)": f"{round(item['prev_other'] * 100)}%",
                "Cluster PMI": f"{item['pmi']:.2f}",
                "p-value": format_p_val(item["p_val"]),
            }
        )

    return payload




# ==========================================
# EXPORT RESULTS TO JSON
# ==========================================
def export_eligible_clusters_to_json(
    df_transformed: pd.DataFrame, 
    patient_lookup: dict, 
    tree_lookup: dict, 
    hdbscan_cols: list, 
    json_filepath: str,
    cohort_archetypes: list = None
):
    """
    Exports all eligible k clustering results into a single consolidated JSON file.
    Each outcome is stored under ordered rank keys (optimal, second, third, etc.).
    Includes patient-level traceability records (DOI, patient label, symptoms list).
    """
    # Ordinal ranking labels for the results dictionary
    rank_labels = [
        "optimal", "second", "third", "fourth", "fifth", 
        "sixth", "seventh", "eighth", "ninth", "tenth"
    ]

    results_by_config = {}

    for idx, col in enumerate(hdbscan_cols):
        # Determine the rank key (falls back to "rank_N" if more than 10 results exist)
        rank_key = rank_labels[idx] if idx < len(rank_labels) else f"rank_{idx + 1}"

        # Calculate number of non-noise clusters (k)
        unique_non_noise_clusters = [c for c in df_transformed[col].unique() if c != -1]
        k_clusters = len(unique_non_noise_clusters)

        # 1. Extract UMAP Coordinates
        umap_cols = sorted(
            [c for c in df_transformed.columns if c.startswith("UMAP_")],
            key=lambda c: int(c.split("_")[1]) if c.split("_")[1].isdigit() else c
        )
        if not umap_cols:
            umap_cols = list(df_transformed.filter(like="UMAP").columns)

        n_total_dims = len(umap_cols)
        target_dim_count = 4 if n_total_dims >= 4 else min(2, n_total_dims)
        selected_cols = umap_cols[:target_dim_count]
        coords_matrix = df_transformed[selected_cols].to_numpy()

        # 2. Extract Gene-Cluster Crosstable
        gene_crosstab_data = get_gene_crosstab_payload(df_transformed, col, patient_lookup)

        # 3. Extract Prevalent Symptoms & Archetypal Signatures
        phenotype_details = get_cluster_phenotype_details(
            df_transformed, patient_lookup, tree_lookup, col
        )

        clusters_payload = []
        unique_clusters = sorted(df_transformed[col].unique())

        for cid in unique_clusters:
            mask = df_transformed[col] == cid
            sub_coords = coords_matrix[mask]
            cluster_patient_ids = df_transformed.loc[mask, "ids"].tolist()
            patient_count = len(cluster_patient_ids)

            cluster_name = "Noise / Unclustered Patients" if cid == -1 else f"Cluster {cid}"

            x_coords = [round(float(val), 4) for val in sub_coords[:, 0]]
            y_coords = [round(float(val), 4) for val in sub_coords[:, 1]] if target_dim_count >= 2 else []

            cid_int = int(cid)
            c_details = phenotype_details.get(cid_int, {})

            # 4. Extract Patient Traceability Records for the Cluster
            patient_records = []
            for pid in cluster_patient_ids:
                p_entry = patient_lookup.get(pid, {})

                # DOI of the first listed article, and the patient's label in it
                first_doi = patient_first_doi(p_entry)
                patients_label = patient_label(p_entry)

                # Extract and format gene(s) as a comma-separated string
                gene_val = p_entry.get("gene", [])
                if isinstance(gene_val, list):
                    gene_str = ", ".join(str(g) for g in gene_val if g)
                elif gene_val is not None:
                    gene_str = str(gene_val)
                else:
                    gene_str = ""

                # Extract unique symptom list
                symptoms_list = sorted(list(set(extract_symptom_codes(p_entry.get("symptoms", [])))))

                patient_records.append({
                    "id": pid,
                    "doi": first_doi,
                    "patients": patients_label,
                    "gene": gene_str,
                    "symptoms": symptoms_list
                })

            payload_item = {
                "cluster_id": cid_int,
                "cluster_name": cluster_name,
                "patient_count": patient_count,
                "x": x_coords,
                "y": y_coords,
                "dimensions_exported": target_dim_count,
                "coordinate_space": df_transformed.attrs.get("coordinate_space", "UMAP"),
                "umap_coordinates": {
                    f"UMAP_{i + 1}": [round(float(val), 4) for val in sub_coords[:, i]]
                    for i in range(target_dim_count)
                },
                "patients": patient_records,
                "prevalent_symptoms": c_details.get("prevalent_symptoms", []),
                "archetypal_symptoms": c_details.get("archetypal_symptoms", []),
                "associated_symptoms": c_details.get("associated_symptoms", [])
            }
            clusters_payload.append(payload_item)

        # Key payload for this specific configuration stored under rank_key
        results_by_config[rank_key] = {
            "is_primary_optimal_result": (idx == 0),
            "num_clusters_k": k_clusters,
            "clustering_column": col,
            "gene_cluster_crosstable": gene_crosstab_data,
            "clusters": clusters_payload
        }

    # Consolidated Single JSON Object
    output_root = {
        "cohort_archetypal_symptoms": cohort_archetypes or [],
        "clustering_results": results_by_config
    }

    # Ensure parent directory exists and write single JSON
    json_path = Path(json_filepath)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "wb") as f:
        f.write(orjson.dumps(output_root, option=orjson.OPT_INDENT_2))

    print(f"All {len(hdbscan_cols)} eligible clustering outcomes exported to single JSON: {json_filepath}")

def get_gene_crosstab_payload(df_transformed: pd.DataFrame, cluster_col: str, patient_lookup: dict = None) -> dict:
    """
    Formats the gene-cluster crosstable into a JSON-serializable dictionary safely.
    """
    valid_df = df_transformed[df_transformed[cluster_col] != -1].copy()
    
    if valid_df.empty:
        return {}

    # Ensure 'genes' column exists
    if "genes" not in valid_df.columns:
        if "ids" in valid_df.columns and patient_lookup:
            valid_df["genes"] = valid_df["ids"].apply(
                lambda pid: patient_lookup.get(pid, {}).get("gene", "Unknown")
            )
        else:
            return {}
            
    def format_gene(g_val):
        if isinstance(g_val, list):
            return ",".join(g_val)
        return str(g_val)

    valid_df["Gene"] = valid_df["genes"].apply(format_gene)

    crosstab = pd.crosstab(
        valid_df["Gene"], 
        valid_df[cluster_col], 
        margins=True, 
        margins_name="Total"
    )
    
    # Format cluster column names as string keys
    crosstab.columns = [f"Cluster_{c}" if isinstance(c, (int, np.integer)) else str(c) for c in crosstab.columns]
    
    return crosstab.to_dict(orient="index")


def get_cluster_phenotype_details(
    df_transformed: pd.DataFrame, 
    patient_lookup: dict, 
    tree_lookup: dict, 
    cluster_col: str
) -> dict:
    """
    (iii) Prevalent symptoms (in descending order of prevalence)
    (iv) Archetypal phenotypic signatures (in descending order of prevalence)
    (v) Rare sub-phenotype enriched flags
    """
    global_patient_ids = df_transformed['ids'].tolist()
    total_cohort_patients = len(global_patient_ids)
    unique_clusters = sorted([c for c in df_transformed[cluster_col].unique() if c != -1])

    # -------------------------------------------------------------------------
    # GLOBAL PRE-CALCULATION FOR SYMPTOM ASSOCIATIONS
    # -------------------------------------------------------------------------
    global_symptom_counts = Counter()
    cluster_patient_map = {}
    cluster_ancestor_counts = {cid: Counter() for cid in unique_clusters}
    target_direct_counts_map = {cid: Counter() for cid in unique_clusters}

    for cid in unique_clusters:
        c_ids = df_transformed[df_transformed[cluster_col] == cid]['ids'].tolist()
        cluster_patient_map[cid] = c_ids
        
        for pid in c_ids:
            p_entry = patient_lookup.get(pid, {})
            direct_symptoms = set(extract_symptom_codes(p_entry.get("symptoms", [])))
            clean_direct = {str(c).replace("HP:", "").zfill(7) for c in direct_symptoms}
            
            target_direct_counts_map[cid].update(clean_direct)
            cluster_ancestor_counts[cid].update(clean_direct)
            global_symptom_counts.update(clean_direct)

    cluster_details = {}

    for cid in unique_clusters:
        cluster_patient_ids = cluster_patient_map[cid]
        num_patients = len(cluster_patient_ids)

        # --- (iii) Prevalent Symptoms ---
        all_symptoms = []
        for pid in cluster_patient_ids:
            p_entry = patient_lookup.get(pid, {})
            all_symptoms.extend(extract_symptom_codes(p_entry.get("symptoms", [])))

        symptom_counts = Counter(all_symptoms)
        symptom_list = []

        for code, count in symptom_counts.items():
            prev = (count / num_patients) * 100
            hpo_code = f"HP:{code}" if not str(code).startswith("HP:") else code
            desc = get_hpo_label(hpo_code, tree_lookup=tree_lookup)
            
            symptom_list.append({
                "hpo_code": hpo_code,
                "count": count,
                "prevalence_percent": round(prev, 2)
            })

        symptom_list.sort(key=lambda x: x["prevalence_percent"], reverse=True)

        # --- (iv) Archetypal Phenotypic Signatures ---
        archetypes_raw = extract_archetypal_phenotypes(
            target_cluster_id=cid,
            df_transformed=df_transformed,
            cluster_col=cluster_col,
            patient_lookup=patient_lookup,
            tree_lookup=tree_lookup,
            threshold=0.50,
            min_depth_from_root=3
        )

        archetype_list = []
        for item in archetypes_raw:
            prev_str = item.get(f"Cluster {cid}", "0%").replace("%", "")
            prev_val = float(prev_str)
            
            cluster_prevalences = {}
            for key, val in item.items():
                if key.startswith("Cluster "):
                    cluster_num = key.replace("Cluster ", "")
                    cluster_prevalences[f"cluster_{cluster_num}"] = float(str(val).replace("%", ""))

            archetype_list.append({
                "hpo_code": item["HPO Code"],
                "prevalence_percent": prev_val,
                "cluster_prevalences": cluster_prevalences,
                "representation_level": item["Representation Level"]
            })
    
        archetype_list.sort(key=lambda x: x["prevalence_percent"], reverse=True)

        # --- (v) Rare Sub-Phenotype Enriched Flags ---
        associated_raw = extract_associated_symptoms(
            target_cluster_id=cid,
            target_direct_counts=target_direct_counts_map[cid],
            global_symptom_counts=global_symptom_counts,
            cluster_patient_map=cluster_patient_map,
            total_cohort_patients=total_cohort_patients,
            tree_lookup=tree_lookup,
            max_prevalence=1.0,
            min_pmi=0.5,
            max_p_val=0.1,
        )

        associated_list = []
        for item in associated_raw:
            associated_list.append({
                "hpo_code": item["HPO Code"],
                "cluster_prevalence": item["Cluster Prev (%)"],
                "rest_cohort_prevalence": item["Rest Cohort Prev (%)"],
                "p_value": item["p-value"],
            })

        cluster_details[int(cid)] = {
            "prevalent_symptoms": symptom_list,
            "archetypal_symptoms": archetype_list,
            "associated_symptoms": associated_list
        }

    return cluster_details


# ==========================================
# 1. PARSE CLI ARGUMENTS
# ==========================================
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Unified Phenome Distance, UMAP and HDBSCAN Pipeline"
    )
    parser.add_argument(
        "--gene", 
        nargs="+", 
        default=["MYO5B"], 
        help="Target gene symbol(s) to filter cohort (Default: MYO5B). Accepts space-separated lists."
    )
    parser.add_argument(
        "-f", "--patient-file",
        type=str,
        default="261013_pytheasDB.json",
        help="Path to the JSON file containing patient data (Default: 261013_pytheasDB.json)"
    )
    parser.add_argument(
        "--trees-file",
        type=str,
        default="hp_trees.json",
        help="Path to the JSON file containing HPO tree structures (Default: hp_trees.json)"
    )
    parser.add_argument(
        "--excluded-doi",
        type=str,
        default="https://doi.org/10.1136/adc.43.232.646",
        help="Optional DOI string to exclude specific patient papers (Default: https://doi.org/10.1136/adc.43.232.646)"
    )
    parser.add_argument(
        "--exclude-uncertain-genetics",
        action="store_true",
        help="Exclude patients with uncertain genetic diagnoses (multiple genes assigned)."
    )
    parser.add_argument(
        "--min-symptoms", 
        type=int, 
        default=5, 
        help="Minimum symptom threshold per patient (Default: 5)"
    )
    parser.add_argument(
        "--digestive", 
        action="store_true", 
        help="Restrict analysis to digestive system only (Default: All organ systems)"
    )
    parser.add_argument(
        "--gridsearch", 
        type=int, 
        choices=[1, 2, 3], 
        default=1, 
        help="HDBSCAN Grid Search Mode: 1 = Vary HDBSCAN parameters dynamically (min_dist from --min-dist, mcs and ms dynamically populated with ms <= mcs); 2 = Vary UMAP min_dist across 10 values from 0.01 to 0.60 (fixed HDBSCAN params scaled dynamically); 3 = Vary UMAP n_neighbors (--n-neighbors) jointly with the mode-1 HDBSCAN grid"
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.0,
        help="UMAP min_dist used in modes 1 and 3 (Default: 0, as UMAP's documentation recommends for clustering; results before 261028 used 0.05). Ignored in mode 2, which varies it."
    )
    parser.add_argument(
        "--stability",
        type=int,
        default=0,
        help="Resampling stability: number of 80%% subsamples per retained clustering (Default: 0 = off; e.g. 50)"
    )
    parser.add_argument(
        "--flag-delta", type=float, default=0.3,
        help="Few-feature flag: prevalence difference counted as discriminating (Default: 0.3)"
    )
    parser.add_argument(
        "--flag-top", type=float, default=0.5,
        help="Few-feature flag: minimum largest prevalence difference (Default: 0.5)"
    )
    parser.add_argument(
        "--flag-max-features", type=int, default=2,
        help="Few-feature flag: flag clusters with at most this many discriminating symptoms (Default: 2)"
    )
    parser.add_argument(
        "--stability-fraction",
        type=float,
        default=0.8,
        help="Share of patients in each stability subsample (Default: 0.8)"
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        nargs="+",
        default=[10, 15, 30],
        help="Mode 3: UMAP n_neighbors values searched (values > N - 1 are dropped; Default: 10 15 30)"
    )
    parser.add_argument(
        "--score-space",
        choices=["umap", "original"],
        default=None,
        help="Space in which configurations are scored for selection: umap = DBCV in the UMAP embedding; original = silhouette on the original distance matrix (Default: umap for modes 1-2, original for mode 3)"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to export the final cluster coordinates in JSON format (Default: 'cluster_visualization.json' inside the output directory)"
    )
    parser.add_argument(
        "--select-terms",
        action="store_true",
        default=os.environ.get("ALPHI_SELECT_TERMS", "0") == "1",
        help="Weight the distances by co-occurrence: terms that co-occur with others more than by chance "
             "(permutation test) count fully, the others with --select-weight (see term_selection.py). "
             "Also switched on by the environment variable ALPHI_SELECT_TERMS=1 (e.g. for grid runners)."
    )
    parser.add_argument("--select-weight", type=float, default=0.01,
                        help="Weight of the terms not selected (Default: 0.01)")
    parser.add_argument("--select-alpha", type=float, default=0.05,
                        help="Family-wise level of the selection test (Default: 0.05)")
    parser.add_argument("--select-permutations", type=int, default=200,
                        help="Permutations of the selection test (Default: 200)")
    parser.add_argument("--select-min-count", type=int, default=None,
                        help="Minimum number of patients with (and without) a feature for it to be tested "
                             "(Default: max(3, 3%% of the cohort))")
    parser.add_argument(
        "--lca",
        action="store_true",
        default=os.environ.get("ALPHI_LCA", "0") == "1",
        help="Check the clustering with a latent class analysis of feature families (see latent_classes.py) and "
             "export the resulting decision as the optimal solution (column HDBSCAN_hybrid). Also ALPHI_LCA=1."
    )
    parser.add_argument("--lca-max-step", type=int, default=2,
                        help="LCA feature families: largest HPO distance (steps) between two terms of a family (Default: 2)")
    parser.add_argument("--lca-resamples", type=int, default=20,
                        help="LCA: subsamples of 80%% of the patients for the stability of the classes (Default: 20)")
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("ALPHI_WORKERS", "8")),
        help="Worker processes for the pairwise distances (Default: 8, or ALPHI_WORKERS; at most the number "
             "of CPU cores). Each worker holds a copy of the HPO tree (about 300 MB): more workers need more "
             "memory and bring little for cohorts of a few hundred patients. Results do not depend on it."
    )
    parser.add_argument(
        "--no-embedding",
        action="store_true",
        help="Cluster the distance matrix itself with HDBSCAN (metric precomputed), without the UMAP "
             "embedding; configurations are then scored by the silhouette on the distance matrix "
             "(--score-space original). No UMAP is run: the coordinates written to the outputs are the "
             "principal coordinates of the distance matrix (columns still named UMAP_i). Grid-search mode 1 only."
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip cluster phenotypic coherence, Kaplan-Meier survival, and Cox PH evaluations."
    )
    return parser.parse_args()


# ==========================================
# 2. MULTIPROCESSING WORKERS & HELPER FUNCS
# ==========================================
def process_pair_batch(pair_batch, patient_lookup, tree_lookup, digestive_only, id_to_term_map=ID_TO_TERM_MAP,
                       term_weights=None):
    pgmodules.set_term_weights(term_weights)     # in each worker process (spawn/forkserver safe)
    records = []
    for id_a, id_b in pair_batch:
        patient_a = patient_lookup.get(id_a)
        patient_b = patient_lookup.get(id_b)

        if not patient_a or not patient_b:
            continue

        dist = dPhenome(
            patient_a,
            patient_b,
            tree_lookup,
            digestive_only=digestive_only,
            id_to_term_map=id_to_term_map
        )

        gene1 = patient_a.get("gene")
        gene2 = patient_b.get("gene")

        if isinstance(gene1, list):
            gene1 = ",".join(gene1)
        if isinstance(gene2, list):
            gene2 = ",".join(gene2)

        records.append({
            "ID1": id_a,
            "ID2": id_b,
            "gene1": gene1,
            "gene2": gene2,
            "d": dist
        })
    return records


def starmap_helper(args):
    return process_pair_batch(*args)

def normalize_doi(doi_val):
    if not doi_val:
        return None
    if isinstance(doi_val, (int, float)):
        doi_val = str(doi_val)
    if isinstance(doi_val, str):
        return doi_val.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
    return None

def select_patient_cohort(patient_list, sel_genes, excluded_doi=None, exclude_uncertain=False, min_symptoms=5):
    selected_ids = []
    norm_excluded_doi = normalize_doi(excluded_doi) if excluded_doi else None

    for entry in patient_list:
        if "gene" not in entry or "symptoms" not in entry:
            continue

        if norm_excluded_doi:
            if norm_excluded_doi in [normalize_doi(d) for d in patient_dois(entry)]:
                continue

        patient_genes = entry["gene"]
        
        if isinstance(patient_genes, list):
            is_multi_gene = len(patient_genes) > 1
            gene_list = patient_genes
        elif isinstance(patient_genes, str):
            is_multi_gene = ("," in patient_genes) or (";" in patient_genes)
            gene_list = [g.strip() for g in patient_genes.replace(";", ",").split(",")] if is_multi_gene else [patient_genes]
        else:
            continue

        if exclude_uncertain and is_multi_gene:
            continue

        if sel_genes.isdisjoint(gene_list):
            continue

        symptoms = set(extract_symptom_codes(entry["symptoms"]))
        if len(symptoms) < min_symptoms:
            continue

        selected_ids.append(entry["id"])
    return selected_ids


def evaluate_cluster_phenotypic_coherence(df_transformed, patient_lookup, tree_lookup, cluster_col, sel_genes, digestive_only=False, X=None):
    print(f"\n==================================================")
    print(f"PHENOTYPIC COHERENCE ANALYSIS FOR: {cluster_col}")
    print(f"==================================================")

    # Filter out noise points (-1)
    valid_df = df_transformed[df_transformed[cluster_col] != -1].copy()

    # GUARD: If HDBSCAN identifies 100% of points as noise (-1), skip evaluation cleanly!
    if valid_df.empty:
        print("  [Note] HDBSCAN identified 100% of cohort points as noise (-1). Skipping evaluation.")
        return

    global_patient_ids = df_transformed['ids'].tolist()
    total_cohort_patients = len(global_patient_ids)
    
    # Create an O(1) index lookup map for the precomputed squareform distance matrix X
    id_to_idx = {pid: i for i, pid in enumerate(global_patient_ids)} if X is not None else {}

    global_symptom_counts = Counter()

    for pid in global_patient_ids:
        p_entry = patient_lookup.get(pid, {})
        s_set = set(extract_symptom_codes(p_entry.get("symptoms", [])))
        global_symptom_counts.update(s_set)

    unique_clusters = sorted([c for c in df_transformed[cluster_col].unique() if c != -1])

    # -------------------------------------------------------------------------
    # PRE-CALCULATION FOR SYMPTOM ASSOCIATIONS
    # -------------------------------------------------------------------------
    cluster_patient_map = {}
    cluster_ancestor_counts = {cid: Counter() for cid in unique_clusters}
    target_direct_counts_map = {cid: Counter() for cid in unique_clusters}

    for cid in unique_clusters:
        c_ids = df_transformed[df_transformed[cluster_col] == cid]['ids'].tolist()
        cluster_patient_map[cid] = c_ids
        
        for pid in c_ids:
            p_entry = patient_lookup.get(pid, {})
            direct_symptoms = set(extract_symptom_codes(p_entry.get("symptoms", [])))
            clean_direct = {str(c).replace("HP:", "").zfill(7) for c in direct_symptoms}
            
            target_direct_counts_map[cid].update(clean_direct)
            cluster_ancestor_counts[cid].update(clean_direct)
            global_symptom_counts.update(clean_direct)

    for cid in unique_clusters:
        cluster_patient_ids = cluster_patient_map[cid]
        num_patients = len(cluster_patient_ids)

        all_cluster_symptoms = []
        patient_symptom_sets = []
        patient_symptom_counts = []
        article_year_map = {}

        for pid in cluster_patient_ids:
            patient_entry = patient_lookup.get(pid, {})
            symptoms = set(extract_symptom_codes(patient_entry.get("symptoms", [])))
            
            all_cluster_symptoms.extend(symptoms)
            patient_symptom_sets.append(symptoms)
            patient_symptom_counts.append(len(symptoms))

            for art in patient_articles(patient_entry):
                single_doi = str(art.get("doi") or "").strip()
                if not single_doi or article_year_map.get(single_doi) is not None:
                    continue
                try:
                    article_year_map[single_doi] = int(art.get("publication_year"))
                except (TypeError, ValueError):
                    article_year_map[single_doi] = None

        article_years = [year for year in article_year_map.values() if year is not None]
        article_years.sort()

        mean_symptoms = np.mean(patient_symptom_counts) if patient_symptom_counts else 0.0
        sd_symptoms = np.std(patient_symptom_counts, ddof=1) if len(patient_symptom_counts) > 1 else 0.0
        num_distinct_dois = len(article_year_map)

        jaccard_scores = []
        intra_distances = []

        for i in range(num_patients):
            pid_a = cluster_patient_ids[i]
            sA = patient_symptom_sets[i]
            patient_a = patient_lookup.get(pid_a)

            for j in range(i + 1, num_patients):
                pid_b = cluster_patient_ids[j]
                sB = patient_symptom_sets[j]
                patient_b = patient_lookup.get(pid_b)

                if sA or sB:
                    jaccard_scores.append(len(sA & sB) / len(sA | sB))

                # O(1) Matrix Lookup if X is provided, otherwise fallback to dPhenome()
                if X is not None and pid_a in id_to_idx and pid_b in id_to_idx:
                    idx_a = id_to_idx[pid_a]
                    idx_b = id_to_idx[pid_b]
                    d_val = X[idx_a, idx_b]
                    intra_distances.append(d_val)
                elif patient_a and patient_b:
                    d_val = dPhenome(patient_a, patient_b, tree_lookup, digestive_only=digestive_only, id_to_term_map=ID_TO_TERM_MAP)
                    intra_distances.append(d_val)

        mean_jaccard = np.mean(jaccard_scores) if jaccard_scores else 0.0
        mean_intra_dist = np.mean(intra_distances) if intra_distances else 0.0
        intra_similarity = 1.0 / (1.0 + mean_intra_dist) if intra_distances else 0.0

        article_years_str = ", ".join(map(str, article_years)) if article_years else "None reported"

        print(f"\nCluster {cid} (N = {num_patients} patients):")
        print(f"  -> Mean (SD) Symptoms per Patient: {mean_symptoms:.2f} ({sd_symptoms:.2f})")
        print(f"  -> Distinct Articles (DOIs): {num_distinct_dois}")
        print(f"  -> Publication Years per Distinct Article (Oldest to Newest): [{article_years_str}]")
        print(f"  -> Phenotypic Coherence Index (Mean Jaccard): {mean_jaccard:.4f}")
        print(f"  -> Mean Intra-Cluster Phenotypic Distance (HPO): {mean_intra_dist:.4f}")
        print(f"  -> Mean Intra-Cluster Phenotypic Similarity (1 / (1 + d)): {intra_similarity:.4f}")

        symptom_counts = Counter(all_cluster_symptoms)
        symptom_stats = []

        for code, c_count in symptom_counts.items():
            p_cluster = c_count / num_patients

            clean_code = str(code).replace("HP:", "").zfill(7)
            g_count = global_symptom_counts.get(clean_code, 0)
            p_global = g_count / total_cohort_patients

            pmi = math.log2(p_cluster / p_global) if p_global > 0 and p_cluster > 0 else 0.0

            hpo_code_formatted = f"HP:{code}" if not str(code).startswith("HP:") else code
            description = get_hpo_label(hpo_code_formatted, tree_lookup=tree_lookup)

            symptom_stats.append({
                "code": code,
                "HPO Code": hpo_code_formatted,
                "Description": description,
                "count": c_count,
                "p_cluster": p_cluster,
                "pmi": pmi,
                "Prevalence (%)": f"{round(p_cluster * 100)}%",
                "Cluster Symptom Coherence (PMI)": f"{pmi:.2f}"
            })

        # Top 10 Ranked by Prevalence
        top_prev = sorted(symptom_stats, key=lambda x: (x["p_cluster"], x["pmi"]), reverse=True)[:10]
        rows_prev = [{
            "HPO Code": s["HPO Code"],
            "Description": s["Description"],
            "Prevalence (%)": s["Prevalence (%)"],
            "Cluster Symptom Coherence (PMI)": s["Cluster Symptom Coherence (PMI)"]
        } for s in top_prev]

        print("\nTop Enriched Symptom Coherence Table (Ranked by Prevalence):")
        print(tabulate(rows_prev, headers="keys", tablefmt="pipe"))

        # Top 10 Ranked by PMI
        pmi_candidates = [s for s in symptom_stats if s["count"] >= 2] or symptom_stats
        top_pmi = sorted(pmi_candidates, key=lambda x: (x["pmi"], x["p_cluster"]), reverse=True)[:10]
        rows_pmi = [{
            "HPO Code": s["HPO Code"],
            "Description": s["Description"],
            "Prevalence (%)": s["Prevalence (%)"],
            "Cluster Symptom Coherence (PMI)": s["Cluster Symptom Coherence (PMI)"]
        } for s in top_pmi]

        print("\nTop Enriched Symptom Coherence Table (Ranked by PMI):")
        print(tabulate(rows_pmi, headers="keys", tablefmt="pipe"))

        # FIXED CALL: Using 'cid' and the precalculated dictionaries
        associated_profile = extract_associated_symptoms(
            target_cluster_id=cid,
            target_direct_counts=target_direct_counts_map[cid],
            global_symptom_counts=global_symptom_counts,
            cluster_patient_map=cluster_patient_map,
            total_cohort_patients=total_cohort_patients,
            tree_lookup=tree_lookup,
            max_prevalence=1.0,
            min_pmi=0.5,
            max_p_val=0.1,
        )

        print(f"\nStrongest statistical associations with cluster {cid}:")
        if associated_profile:
            print(tabulate(associated_profile, headers="keys", tablefmt="pipe"))
        else:
            print("  [Note] No significant associations to report.")

        # Archetypal Phenotypic Signature (Shared by >= 50% via parent_branches DAG Rollup)
        archetype_results = extract_archetypal_phenotypes(
            target_cluster_id=cid,
            df_transformed=df_transformed,
            cluster_col=cluster_col,
            patient_lookup=patient_lookup,
            tree_lookup=tree_lookup,
            threshold=0.50,
            min_depth_from_root=3
        )

        print(f"\nArchetypal Phenotypic Signature (Shared by ≥50% of Cluster {cid} via DAG Rollup and <50% in other clusters):")
        if archetype_results:
            print(tabulate(archetype_results, headers="keys", tablefmt="pipe"))
        else:
            print("  [Note] No direct or aggregated ancestor terms reached the ≥50%, <50% thresholds.")
    
    if len(sel_genes) > 1:
        print(f"\nGene x Cluster Crosstable ({cluster_col}):")
        
        valid_df = df_transformed[df_transformed[cluster_col] != -1].copy()
        
        # Guard: Re-populate 'genes' from patient_lookup if missing
        if "genes" not in valid_df.columns:
            if "ids" in valid_df.columns:
                valid_df["genes"] = valid_df["ids"].apply(
                    lambda pid: patient_lookup.get(pid, {}).get("gene", "Unknown")
                )
            else:
                print("  [Warning] Neither 'genes' nor 'ids' column present. Skipping crosstable.")
                return

        def is_unambiguous_gene(g_val):
            if isinstance(g_val, list):
                return len(g_val) == 1
            if isinstance(g_val, str):
                return "," not in g_val and ";" not in g_val
            return False

        # Filter for unambiguous genes using the guaranteed 'genes' column
        valid_df = valid_df[valid_df["genes"].apply(is_unambiguous_gene)].copy()
        
        valid_df["Gene"] = valid_df["genes"].apply(
            lambda g: g[0] if isinstance(g, list) else str(g)
        )
        crosstab = pd.crosstab(
            valid_df["Gene"], 
            valid_df[cluster_col], 
            margins=True, 
            margins_name="Total"
        )
        
        crosstab.columns = [f"Cluster {c}" if isinstance(c, (int, np.integer)) else c for c in crosstab.columns]
        crosstab = crosstab.reset_index()

        gene_table_kable = tabulate(crosstab, headers="keys", tablefmt="pipe", showindex=False)
        print(gene_table_kable)


def select_optimal_clustering_configurations(df_grid: pd.DataFrame, lambda_penalty: float = 0.3,
                                             score_col: str = "dbcv_score"):
    """
    Executes a robust multi-metric ranking and selection algorithm on HDBSCAN grid search results.
    Adaptively switches base scoring to point_coverage if >50% of runs have NaN DBCV scores.
    score_col: the validity score used ("dbcv_score" = DBCV in the UMAP embedding;
    "silhouette_original" = silhouette on the original distance matrix, --score-space original).
    """
    print("\n" + "=" * 80)
    print("       STARTING ROBUST SELECTION ALGORITHM (ADAPTIVE DBCV / COVERAGE)        ")
    print("=" * 80)
    
    df_grid_clean = df_grid[df_grid["num_clusters"].notna()].copy()
    if df_grid_clean.empty:
        raise ValueError("No valid clustering configurations found.")
    
    total_runs = len(df_grid_clean)
    print(f"[Validity score] {score_col}")
    nan_dbcv_count = df_grid_clean[score_col].isna().sum()
    nan_dbcv_ratio = nan_dbcv_count / total_runs if total_runs > 0 else 0.0
    
    print(f"[Step 0] Total evaluated grid configurations: {total_runs}")
    print(f"         -> NaN DBCV count: {nan_dbcv_count} ({nan_dbcv_ratio:.1%} of runs)")
    
    # Adaptive Trigger: If >50% of runs have NaN DBCV, switch base scoring to point_coverage
    use_coverage_base = nan_dbcv_ratio > 0.5
    if use_coverage_base:
        print("\n[Adaptive Fallback Triggered] >50% of runs have NaN DBCV scores.")
        print("                            Switching base score foundation from DBCV to point_coverage.")
    
    unique_k_count = df_grid_clean["num_clusters"].nunique()
    is_coarse_no_consensus = (unique_k_count == total_runs) and (total_runs > 1)
    
    if "noise_ratio" not in df_grid_clean.columns and "point_coverage" in df_grid_clean.columns:
        df_grid_clean["noise_ratio"] = 1.0 - df_grid_clean["point_coverage"]
    elif "noise_ratio" not in df_grid_clean.columns:
        df_grid_clean["noise_ratio"] = 0.0
        df_grid_clean["point_coverage"] = 1.0

    df_grid_clean["dbcv_score_filled"] = df_grid_clean[score_col].fillna(0.0)
    
    base_scores = []
    p_insts = []
    final_scores = []
    confidence_flags = []
    
    for idx, row in df_grid_clean.iterrows():
        k = row["num_clusters"]
        coverage = row["point_coverage"]
        
        if use_coverage_base:
            # Adaptive Base Score: Reward coverage directly for k=1 or low-DBCV regimes
            if k == 1:
                base_score = coverage  # Fairly score parsimonious coverage without arbitrary 0 penalty
            else:
                dbcv = row["dbcv_score_filled"]
                base_score = coverage * (0.5 + 0.5 * max(0.0, dbcv))
        else:
            # Standard DBCV-driven base score for multi-cluster dense regimes
            if k < 2:
                base_score = 0.0
            else:
                dbcv = row["dbcv_score_filled"]
                noise_ratio = row["noise_ratio"]
                base_score = dbcv * (1.0 - noise_ratio)
                
        base_scores.append(base_score)
        
        if is_coarse_no_consensus:
            p_inst = 0.0
            final_score = base_score - (0.1 * k) 
            flag = "UNCERTAIN (Coarse Grid Fallback)"
        else:
            p_inst = evaluate_local_micro_stability(row)
            final_score = base_score - (lambda_penalty * p_inst)
            flag = "STABLE" if p_inst <= 1.0 else "UNSTABLE"
            
        p_insts.append(p_inst)
        final_scores.append(final_score)
        confidence_flags.append(flag)
        
    df_grid_clean["base_score"] = base_scores
    df_grid_clean["instability_penalty"] = p_insts
    df_grid_clean["final_score"] = final_scores
    df_grid_clean["confidence_flag"] = confidence_flags
    
    max_final = df_grid_clean["final_score"].max() if not df_grid_clean["final_score"].empty else 0.0
    max_coverage = df_grid_clean["point_coverage"].max()
    
    score_cutoff = 0.75 * max_final if max_final > 0 else 0.0
    coverage_cutoff = 0.80 * max_coverage
    
    df_valid = df_grid_clean[
        (df_grid_clean["final_score"] >= score_cutoff) | 
        (df_grid_clean["point_coverage"] >= coverage_cutoff) |
        (df_grid_clean["num_clusters"] < 2)
    ].copy()
    
    unique_ks = df_valid["num_clusters"].unique()
    selected_per_k = []
    
    print(f"\n[Step 2] Extracting Top Representative per Distinct Value of k:")
    for k in sorted(unique_ks):
        df_k = df_valid[df_valid["num_clusters"] == k].copy()
        top_k_row = df_k.sort_values(
            by=["final_score", "point_coverage"], 
            ascending=[False, False], 
            na_position="last"
        ).iloc[0]
        selected_per_k.append(top_k_row)
        
        config_id = top_k_row.get(
            "col_name", 
            f"mcs_{top_k_row.get('min_cluster_size', 'NA')}_ms_{top_k_row.get('min_samples', 'NA')}"
        )
        print(
            f"         - k = {k}: Selected config '{config_id}' "
            f"(FinalScore: {top_k_row['final_score']:.4f}, Flag: {top_k_row['confidence_flag']})"
        )
        
    df_selected = pd.DataFrame(selected_per_k)
    
    if is_coarse_no_consensus:
        top_configs_df = df_selected.sort_values(
            by=["num_clusters", "final_score"], 
            ascending=[True, False], 
            na_position="first"
        ).reset_index(drop=True)
    else:
        top_configs_df = df_selected.sort_values(
            by=["final_score", "point_coverage"], 
            ascending=[False, False], 
            na_position="last"
        ).reset_index(drop=True)
    
    print(f"\n[Step 3] Final Global Robustness Ranking:")
    for idx, row in top_configs_df.iterrows():
        rank_label = "PRIMARY OPTIMAL" if idx == 0 else f"Alternative Rank {idx + 1}"
        config_id = row.get(
            "col_name", 
            f"mcs_{row.get('min_cluster_size', 'NA')}_ms_{row.get('min_samples', 'NA')}"
        )
        print(
            f"         - [{rank_label}] Rank {idx + 1} | k = {row['num_clusters']}: '{config_id}' "
            f"| FinalScore: {row['final_score']:.4f} | Status: {row['confidence_flag']}"
        )
        
    print("\n" + "-" * 80)
    print(f"SELECTION COMPLETE: Retained {len(top_configs_df)} configuration(s). Status flag assigned.")
    print("-" * 80 + "\n")
    
    return top_configs_df, df_grid_clean



def evaluate_local_micro_stability(row) -> float:
    """
    Dynamically probes the immediate micro-neighborhood (mcs ± 1, ms ± 1) 
    using the row's pre-computed UMAP embedding ('coord_matrix') to calculate 
    a true local instability penalty (P_inst) via standard deviation of k.
    """
    base_mcs = row['min_cluster_size']
    base_ms = row['min_samples']
    embedding = row['coord_matrix']  # The UMAP projection matrix for this run
    metric = row.get('hdb_metric', 'euclidean') if hasattr(row, 'get') else 'euclidean'
    if metric == 'precomputed':
        embedding = row['hdb_input']     # --no-embedding: the distance matrix itself
    
    micro_ks = []
    
    # Test micro-deltas (-1, 0, +1)
    for d_mcs in [-1, 0, 1]:
        for d_ms in [-1, 0, 1]:
            test_mcs = base_mcs + d_mcs
            test_ms = base_ms + d_ms
            
            # HDBSCAN structural constraints
            if test_mcs < 2 or test_ms < 1 or test_ms > test_mcs:
                continue
                
            # Run HDBSCAN locally on the existing embedding (extremely fast)
            clusterer = hdbscan.HDBSCAN(min_cluster_size=test_mcs, min_samples=test_ms, metric=metric)
            labels = clusterer.fit_predict(embedding)
            
            # Count valid clusters (excluding noise label -1)
            k = len(set(labels)) - (1 if -1 in labels else 0)
            micro_ks.append(k)
            
    # Calculate local standard deviation as the true instability penalty
    if len(micro_ks) > 1:
        p_inst = float(np.std(micro_ks))
    else:
        p_inst = 0.0
        
    return 0.0 if np.isnan(p_inst) else p_inst

def flag_few_feature_clusters(labels, symptom_sets, delta=0.3, top=0.5, max_features=2):
    """
    Few-feature split check, PAIRWISE: for every pair of non-noise clusters, the symptoms
    whose prevalence differs by >= delta between the two. A pair is flagged when at most
    max_features symptoms differ that much and the largest difference is >= top; a cluster
    is flagged when it belongs to a flagged pair. Pairwise, so it also catches one true
    subgroup cut in two by an isolated symptom (the two halves differ by that symptom only,
    even though each still differs from the other subgroups by many).
    Returns {cluster: {"flag", "n_features" (fewest over its pairs), "partner", "top"}};
    top lists (code, prevalence in the cluster, prevalence in the partner), largest first.
    """
    labels = np.asarray(labels)
    clusters = sorted(set(labels) - {-1})
    out = {}
    if len(clusters) < 2:
        return out
    prev = {}
    for c in clusters:
        members = [symptom_sets[i] for i in range(len(labels)) if labels[i] == c]
        cnt = {}
        for s_ in members:
            for t in s_:
                cnt[t] = cnt.get(t, 0) + 1
        prev[c] = {t: v / len(members) for t, v in cnt.items()}
    for c in clusters:
        best = None
        for o in clusters:
            if o == c:
                continue
            diffs = sorted(((abs(prev[c].get(t, 0) - prev[o].get(t, 0)), t, prev[c].get(t, 0), prev[o].get(t, 0))
                            for t in set(prev[c]) | set(prev[o])), reverse=True)
            n_disc = sum(d[0] >= delta for d in diffs)
            flag = n_disc >= 1 and n_disc <= max_features and diffs[0][0] >= top
            key = (not flag, n_disc)
            if best is None or key < best[0]:
                best = (key, o, flag, n_disc, diffs[:3])
        _, o, flag, n_disc, top3 = best
        out[int(c)] = {"flag": bool(flag), "n_features": int(n_disc), "partner": int(o),
                       "top": [(t, round(a, 3), round(b, 3)) for _, t, a, b in top3]}
    return out


def evaluate_cluster_stability(X, row, clustering_dims, n_resamples, fraction=0.8, seed=1, no_embedding=False):
    """
    Resampling stability (Hennig's clusterboot, subsetting scheme) of one clustering
    configuration. X: full square distance matrix. Returns
    ({cluster label: {"jaccard_mean", "dissolved", "n_resamples"}}, Counter of k over resamples).
    """
    from collections import Counter as _Counter
    labels = np.asarray(row["labels"])
    n = len(labels)
    m = max(2, int(round(fraction * n)))
    rng = np.random.default_rng(seed)
    clusters = sorted(set(labels) - {-1})
    jac = {c: [] for c in clusters}
    k_counts = _Counter()
    for b in range(n_resamples):
        idx = np.sort(rng.choice(n, size=m, replace=False))
        Xs = X[np.ix_(idx, idx)]
        if no_embedding:
            sub = hdbscan.HDBSCAN(min_cluster_size=int(row["min_cluster_size"]), min_samples=int(row["min_samples"]),
                                  cluster_selection_epsilon=0.0, metric="precomputed",
                                  allow_single_cluster=True).fit_predict(Xs.astype(np.float64))
        else:
            nn = int(min(row["umap_n_neighbors"], m - 1))
            emb = UMAP(n_components=clustering_dims, metric="precomputed", n_neighbors=max(2, nn),
                       min_dist=row["umap_min_dist"], random_state=25 + b).fit_transform(Xs)
            sub = hdbscan.HDBSCAN(min_cluster_size=int(row["min_cluster_size"]), min_samples=int(row["min_samples"]),
                                  cluster_selection_epsilon=0.0, metric="euclidean",
                                  allow_single_cluster=True).fit_predict(emb)
        k_counts[len(set(sub) - {-1})] += 1
        found = {d: set(idx[sub == d]) for d in set(sub) - {-1}}
        for c in clusters:
            orig = set(idx[labels[idx] == c])
            if not orig:
                continue
            best = max((len(orig & f) / len(orig | f) for f in found.values()), default=0.0)
            jac[c].append(best)
    out = {}
    for c in clusters:
        v = jac[c]
        out[int(c)] = {"jaccard_mean": float(np.mean(v)) if v else float("nan"),
                       "dissolved": float(np.mean([x <= 0.5 for x in v])) if v else float("nan"),
                       "n_resamples": len(v)}
    return out, k_counts


# ==========================================
# 3. MAIN EXECUTION PIPELINE
# ==========================================
def main():
    args = parse_arguments()
    start_time = time.time()

    fname_patients = args.patient_file
    fname_trees = resolve_resource(args.trees_file)
    excluded_doi = args.excluded_doi
    exclude_uncertain = args.exclude_uncertain_genetics
    
    sel_genes = set(args.gene)
    digestive_only = args.digestive

    date_str = datetime.now().strftime("%y%m%d")
    system_tag = "DIGESTIVE" if digestive_only else "ALL"
    
    # Switch output folder representation to gene first-letters if > 4 genes
    sorted_genes = sorted(sel_genes)
    if len(sorted_genes) > 4:
        gene_tag = "".join(g[0] for g in sorted_genes)
    else:
        gene_tag = "_".join(sorted_genes)
    
    print(f"Reading patient records from: {fname_patients}...")
    try:
        with open(fname_patients, "rb") as f:
            raw_data = orjson.loads(f.read())
            
        # Support both flat list structures and nested dict structures (e.g., synthetic_patients.json)
        if isinstance(raw_data, dict) and "patients" in raw_data:
            patient_list = raw_data["patients"]
        elif isinstance(raw_data, list):
            patient_list = raw_data
        else:
            raise ValueError(
                f"Unexpected JSON structure in '{fname_patients}'. "
                "Expected a list of patients or a dictionary containing a 'patients' key."
            )
            
    except FileNotFoundError:
        print(f"Error: Patient data file '{fname_patients}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing patient data file '{fname_patients}': {e}")
        sys.exit(1)

    patient_lookup = {patient["id"]: patient for patient in patient_list}
    
    print(f"Reading HPO tree structures from: {fname_trees}...")
    try:
        with open(fname_trees, "rb") as f:
            hp_tree_list = orjson.loads(f.read())
    except FileNotFoundError:
        print(f"Error: HPO tree file '{fname_trees}' not found.")
        sys.exit(1)

    tree_lookup = {}
    for branch in hp_tree_list:
        code_raw = str(branch.get("code", "")).replace("HP:", "").zfill(7)
        tree_lookup[code_raw] = branch

    print(f"\nFiltering active patient cohort for gene(s): {', '.join(sorted(sel_genes))} (min symptoms >= {args.min_symptoms})...")
    if excluded_doi:
        print(f"Excluding records matching DOI: {excluded_doi}")
    else:
        print("No DOI exclusions applied.")

    if exclude_uncertain:
        print("Excluding patients with uncertain genetic diagnoses (multiple genes assigned).")

    selected_ids = select_patient_cohort(
        patient_list, 
        sel_genes, 
        excluded_doi=excluded_doi, 
        exclude_uncertain=exclude_uncertain,
        min_symptoms=args.min_symptoms
    )
    n_cohort = len(selected_ids)
    print(f"Cohort filtering complete. Selected {n_cohort} matching patients.")
    check_patient_fields(patient_list, selected_ids, excluded_doi)

    pair_combinations = list(combinations(selected_ids, 2))
    total_combos = len(pair_combinations)

    if total_combos == 0:
        print("[Halted] No pairwise combinations met the evaluation criteria.")
        sys.exit(0)

    # Compute Dynamic Parameters with Quantized Step Rules
    dyn = compute_dynamic_parameters(n_cohort)
    clustering_dims = dyn["n_components"]
    n_neighbors_dyn = dyn["n_neighbors"]

    print(f"\n[Dynamic Cohort Scaling Activated for N={n_cohort} Patients]")
    print(f"  -> UMAP Manifold Dimensions (n_components) : {clustering_dims}")
    print(f"  -> UMAP Neighborhood Size (n_neighbors)    : {n_neighbors_dyn}")
    print(f"  -> HDBSCAN min_cluster_size Search Space   : {dyn['min_cluster_sizes']}")
    print(f"  -> HDBSCAN min_samples Search Space        : {dyn['min_samples_list']}")

    # Use pathlib.Path for safer, modern path construction
    result_folder = Path(f"{date_str}_{gene_tag}_{system_tag}_{total_combos}")
    result_folder.mkdir(parents=True, exist_ok=True)

    # Copy current running script into the output folder for traceability
    script_path = Path(__file__).resolve()
    copied_script_path = result_folder / script_path.name
    shutil.copy2(script_path, copied_script_path)

    log_file_path = result_folder / "pipeline.log"
    tee_logger = TeeLogger(str(log_file_path))  # Pass as string if TeeLogger expects a string path
    sys.stdout = tee_logger
    sys.stderr = tee_logger
    
    # -------------------------------------------------------------------------
    # PRINT EXACT COMMAND INVOCATION AT TOP OF LOG FILE
    # -------------------------------------------------------------------------
    import shlex
    command_str = f"{sys.executable} " + " ".join(shlex.quote(arg) for arg in sys.argv)
    print("================================================================================")
    print(f"COMMAND EXECUTED: {command_str}")
    print("================================================================================\n")

    print(f"Allocating outputs and execution log to directory: {result_folder}")

    # Multiprocessing Distance Calculation
    num_cores = max(1, min(args.workers, mp.cpu_count()))
    chunk_size = math.ceil(total_combos / num_cores)
    batches = [
        pair_combinations[i : i + chunk_size]
        for i in range(0, total_combos, chunk_size)
    ]

    print(f"\nDistributing {total_combos} pair distance calculations across {num_cores} CPU cores...")
    print(f"Organ Scope: {'Digestive System Only' if digestive_only else 'All Organ Systems'}")
    
    # Optional selection of co-occurring terms (term_selection.py)
    term_weights = None
    if args.select_terms:
        code_sets = [{norm_code(c) for c in extract_symptom_codes(patient_lookup[sid].get("symptoms", []))}
                     for sid in selected_ids]
        term_weights, sel_report = select_terms(code_sets, tree_lookup, n_perm=args.select_permutations,
                                                alpha=args.select_alpha, min_count=args.select_min_count,
                                                low_weight=args.select_weight)
        print(f"\n[Term selection] {sel_report['n_terms']} terms, {sel_report['n_tested']} features tested "
              f"(present in {sel_report['min_count']} to N - {sel_report['min_count']} patients); "
              f"threshold Q > {sel_report['threshold'] if sel_report['threshold'] is not None else float('nan'):.1f} "
              f"({sel_report['n_perm']} permutations, alpha {sel_report['alpha']})")
        if sel_report["selected"]:
            n_full = sum(1 for w in term_weights.values() if w == 1.0)
            print(f"  {len(sel_report['selected'])} features selected; {n_full} of {len(term_weights)} terms "
                  f"weighted 1, the others {args.select_weight}:")
            for t in sel_report["selected"]:
                print(f"    HP:{t}  {get_hpo_label(t, tree_lookup)}  Q = {sel_report['q'][t]:.1f}")
        else:
            print("  Fewer than 2 features selected: no co-occurrence beyond chance; all terms weighted 1.")
            term_weights = None
        with open(result_folder / "term_selection.json", "w", encoding="utf-8") as fh:
            json.dump({k: v for k, v in sel_report.items()} | {"weights": term_weights}, fh, indent=1)
    pgmodules.set_term_weights(term_weights)   # main process; also clears weights left by an earlier in-process run

    pool_args = [
        (batch, patient_lookup, tree_lookup, digestive_only, ID_TO_TERM_MAP, term_weights)
        for batch in batches
    ]

    all_results = []
    with mp.Pool(processes=num_cores) as pool:
        for batch_res in tqdm(
            pool.imap(starmap_helper, pool_args), 
            total=len(pool_args), 
            desc="Calculating Pairwise Distances", 
            unit="batch",
            file=tee_logger.terminal
        ):
            all_results.extend(batch_res)

    cdmdf = pd.DataFrame(all_results)

    cdm_file = result_folder / "cdm.csv"
    cdmdf.to_csv(cdm_file, index=False)
    print(f"Condensed distance matrix exported to: {cdm_file}")
    
    # Dynamic UMAP & HDBSCAN Grid Search
    print(f"\nPreparing squareform matrix for {clustering_dims}D UMAP reduction...")
    X = squareform(cdmdf["d"].to_numpy())

    ids = pd.Series(selected_ids)
    symptomslist = [len(set(extract_symptom_codes(patient_lookup[sid]["symptoms"]))) for sid in ids]
    doilist = [patient_first_doi(patient_lookup[sid]) for sid in ids]
    geneslist = [patient_lookup[sid].get("gene", None) for sid in ids]

    if args.no_embedding and args.gridsearch != 1:
        print("[Halted] --no-embedding is available with --gridsearch 1 only.")
        sys.exit(1)
    if args.no_embedding and args.score_space == "umap":
        print("[Note] --no-embedding: no UMAP space to score in; using --score-space original.")
    score_space = "original" if args.no_embedding else (args.score_space or ("original" if args.gridsearch == 3 else "umap"))
    score_col = "silhouette_original" if score_space == "original" else "dbcv_score"
    print(f"\n[Selection score] {score_space} space ({score_col})")

    param_grid = []          # (n_neighbors, min_dist, mcs, ms)
    if args.gridsearch == 3:
        nn_list = sorted({v for v in args.n_neighbors if 2 <= v <= dyn["max_neighbors"]}) or [n_neighbors_dyn]
        print(f"\n[Gridsearch Mode 3] Joint UMAP n_neighbors x HDBSCAN variation (min_dist fixed at {args.min_dist}, "
              f"n_neighbors in {nn_list}, mcs in {dyn['min_cluster_sizes']} and ms in {dyn['min_samples_list']} with ms <= mcs).")
        for nn in nn_list:
            for mcs in dyn["min_cluster_sizes"]:
                for ms in dyn["min_samples_list"]:
                    if ms <= mcs:
                        param_grid.append((nn, args.min_dist, mcs, ms))
    elif args.gridsearch == 1:
        md = args.min_dist
        print(f"\n[Gridsearch Mode 1] Dynamic HDBSCAN parameter variation (min_dist fixed at {md}, mcs in {dyn['min_cluster_sizes']} and ms in {dyn['min_samples_list']} with ms <= mcs).")
        for mcs in dyn["min_cluster_sizes"]:
            for ms in dyn["min_samples_list"]:
                if ms <= mcs:
                    param_grid.append((n_neighbors_dyn, md, mcs, ms))
    else:
        def_mcs = dyn["default_mcs"]
        def_ms = dyn["default_ms"]
        print(f"\n[Gridsearch Mode 2] UMAP min_dist parameter variation (HDBSCAN fixed dynamically at mcs={def_mcs}, ms={def_ms}).")
        min_dist_opts = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6]
        for md in min_dist_opts:
            param_grid.append((n_neighbors_dyn, md, def_mcs, def_ms))

    grid_results = []
    
    total_grid_runs = len(param_grid)
    grid_pbar = tqdm(total=total_grid_runs, desc=f"Grid Search ({clustering_dims}D UMAP Space)", unit="eval", file=tee_logger.terminal)

    grouped_params = {}
    for nn, md, mcs, ms in param_grid:
        grouped_params.setdefault((nn, md), []).append((mcs, ms))

    total_points = len(X)    

    X_precomputed = np.asarray(X, dtype=np.float64)
    for (nn, md), hdb_params in grouped_params.items():
        if args.no_embedding:
            X_clustering = None                        # HDBSCAN on the distance matrix itself
        else:
            embedding_cluster = UMAP(
                n_components=clustering_dims,
                metric="precomputed",
                n_neighbors=nn,
                min_dist=md,
                random_state=25
            )
            X_clustering = embedding_cluster.fit_transform(X)

        for mcs, ms in hdb_params:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=mcs,
                min_samples=ms,
                cluster_selection_epsilon=0.0,
                metric="precomputed" if args.no_embedding else "euclidean",
                allow_single_cluster=True,
                gen_min_span_tree=True
            )
            labels = clusterer.fit_predict(X_precomputed if args.no_embedding else X_clustering)

            num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_points = np.sum(labels == -1)

            # Calculate coverage (1.0 = 100% clustered, 0.0 = 100% noise)
            point_coverage = (total_points - noise_points) / total_points
            
            unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
            cluster_dist = ", ".join(map(str, counts)) if len(counts) > 0 else "None"

            dbcv = np.nan if args.no_embedding else compute_dbcv(X_clustering, labels)

            if num_clusters > 1 and not args.no_embedding:
                try:
                    mask = labels != -1
                    sil = silhouette_score(X_clustering[mask], labels[mask], metric="euclidean")
                except:
                    sil = np.nan
            else:
                sil = np.nan

            # silhouette on the original distance matrix (same space for every embedding)
            sil_orig = np.nan
            if num_clusters > 1:
                mask = labels != -1
                if 2 <= len(set(labels[mask])) <= mask.sum() - 1:
                    try:
                        sil_orig = float(silhouette_score(X[np.ix_(mask, mask)], labels[mask], metric="precomputed"))
                    except Exception:
                        sil_orig = np.nan

            grid_results.append({
                "umap_n_neighbors": nn,
                "umap_min_dist": md,
                "min_cluster_size": mcs,
                "min_samples": ms,
                "silhouette_score": sil,
                "silhouette_original": sil_orig,
                "dbcv_score": dbcv,
                "point_coverage": point_coverage,
                "num_clusters": num_clusters,
                "noise_points": noise_points,
                "cluster_distribution": cluster_dist,
                "labels": labels.tolist(),
                "coord_matrix": X_clustering,
                "hdb_metric": "precomputed" if args.no_embedding else "euclidean",
                "hdb_input": X_precomputed if args.no_embedding else None
            })
            grid_pbar.update(1)

    grid_pbar.close()

    df_grid = pd.DataFrame(grid_results)
    df_grid["col_name"] = df_grid.apply(
            lambda r: (f"nn_{r['umap_n_neighbors']}_" if args.gridsearch == 3 else "")
                      + f"umd_{r['umap_min_dist']}_mcs_{r['min_cluster_size']}_ms_{r['min_samples']}", 
            axis=1
        )

    # --- SELECTION ALGORITHM & ROBUSTNESS SCORING ---
    top_configs_df, df_grid_scored = select_optimal_clustering_configurations(df_grid, score_col=score_col)

    # Sort the full grid search results by the robust final score descending
    df_grid_scored = df_grid_scored.sort_values(by="final_score", ascending=False).reset_index(drop=True)

    print(f"\n=== GRID SEARCH RESULTS ({clustering_dims}D SPACE - SORTED BY ROBUST FINAL SCORE) ===")
    df_display = df_grid_scored.drop(columns=["labels", "coord_matrix", "hdb_metric", "hdb_input", "base_score", "col_name", "noise_ratio", "confidence_flag"], errors="ignore")
    print(tabulate(df_display, headers="keys", tablefmt="pipe", showindex=False, floatfmt=".4f"))

    # --- FINAL OPTIMAL CONFIGURATIONS SUMMARY TABLE ---
    print(f"\n=== FINAL GLOBAL ROBUSTNESS RANKING TABLE ===")
    cols_to_display = [c for c in ['num_clusters', 'col_name', 'final_score', 'dbcv_score', 'silhouette_score', 'silhouette_original', 'point_coverage', 'instability_penalty', 'confidence_flag'] if c in top_configs_df.columns]
    print(tabulate(top_configs_df[cols_to_display], headers="keys", tablefmt="pipe", showindex=False, floatfmt=".4f"))

    # --- DOWNSTREAM PROCESSING ---
    # Explicitly identify the globally optimal configuration based on the highest final_score,
    # independent of any display sorting order in top_configs_df (e.g., coarse grid fallback sorting).
    best_config_row = top_configs_df.sort_values(by="final_score", ascending=False).iloc[0]
    best_X_clustering = best_config_row["coord_matrix"]
    if args.no_embedding:
        # coordinates for the outputs only (the clusters come from the distance matrix):
        # principal coordinates of the distance matrix, no UMAP
        best_X_clustering = principal_coordinates(X, clustering_dims)
        print(f"[No embedding] Output coordinates: first {clustering_dims} principal coordinates of the distance "
              f"matrix (columns and JSON keys keep the names UMAP_i for compatibility).")

    df_transformed = pd.DataFrame(
        best_X_clustering, 
        columns=[f"UMAP_{i+1}" for i in range(clustering_dims)]
    )
    df_transformed.attrs["coordinate_space"] = "principal coordinates" if args.no_embedding else "UMAP"

    df_transformed = df_transformed.assign(
        ids=ids, 
        genes=geneslist, 
        doi=doilist, 
        symptoms=symptomslist
    )

    primary_cluster_col = None
    stability_results = {}
    flag_results = {}
    symptom_sets = [set(extract_symptom_codes(patient_lookup[sid]["symptoms"])) for sid in ids]
    if args.stability > 0:
        print(f"\n[Stability] {args.stability} subsamples of {args.stability_fraction:.0%} of the patients "
              f"per retained clustering ({len(top_configs_df)} clustering(s))...")
    for idx, row in top_configs_df.iterrows():
        if args.gridsearch == 1:
            col_name = f"HDBSCAN_mcs{row['min_cluster_size']}_ms{row['min_samples']}"
        elif args.gridsearch == 3:
            col_name = f"HDBSCAN_nn{row['umap_n_neighbors']}_mcs{row['min_cluster_size']}_ms{row['min_samples']}"
        else:
            col_name = f"HDBSCAN_md{row['umap_min_dist']}_k{row['num_clusters']}"

        df_transformed[col_name] = row["labels"]
        
        # Ensure primary_cluster_col locks onto the true top-scoring configuration row
        if row.equals(best_config_row):
            primary_cluster_col = col_name

        flag_results[col_name] = (flag_few_feature_clusters(
            row["labels"], symptom_sets, args.flag_delta, args.flag_top, args.flag_max_features),
            int(row["num_clusters"]))

        if args.stability > 0:
            stab, kc = evaluate_cluster_stability(X, row, clustering_dims, args.stability,
                                                  fraction=args.stability_fraction,
                                                  no_embedding=args.no_embedding)
            stability_results[col_name] = (stab, kc, int(row["num_clusters"]))
     
    # -------------------------------------------------------------------------
    # LATENT CLASS CHECK (--lca): decision exported as the optimal solution
    # -------------------------------------------------------------------------
    if args.lca:
        lca_sets = [{norm_code(c) for c in extract_symptom_codes(patient_lookup[sid]["symptoms"])} for sid in ids]
        a_stab = stability_results.get(primary_cluster_col, (None,))[0] if stability_results else None
        lrep = latent_classes.check(
            lca_sets, tree_lookup, df_transformed[primary_cluster_col].to_numpy(),
            flag_fn=lambda lab: flag_few_feature_clusters(lab, symptom_sets, args.flag_delta, args.flag_top,
                                                          args.flag_max_features),
            alphi_flags=flag_results.get(primary_cluster_col, ({},))[0], alphi_stability=a_stab,
            max_step=args.lca_max_step, n_resamples=args.lca_resamples)
        print("\n=== LATENT CLASS CHECK (feature families, BIC) ===")
        print(f"  {lrep['n_families']} feature families; ALPHI: {lrep['k_alphi']} cluster(s); "
              f"LCA: {lrep['k_lca']} class(es)" + (f", sizes {lrep.get('class_sizes')}" if lrep.get("class_sizes") else ""))
        if "bic" in lrep:
            print(f"  BIC by number of classes: {lrep['bic']}")
            print(f"  ARI between ALPHI and LCA (patients assigned by ALPHI): {lrep['ari_alphi_lca']:.3f}")
            print(f"  LCA class stability (mean Jaccard): " +
                  ", ".join(f"{c}: {v:.2f}" for c, v in lrep["lca_stability"].items()) +
                  (f"; few-feature flags: {lrep['lca_flagged']}" if lrep["lca_flagged"] else ""))
        print(f"  Decision: {lrep['decision']}")
        print(f"  Patients placed differently by ALPHI and LCA: {len(lrep['disagreements'])}")
        for d in lrep["disagreements"][:50]:
            print(f"    {ids.iloc[d['index']]}: ALPHI cluster {d['alphi_cluster']}, LCA class {d['lca_class']} "
                  f"(mostly ALPHI cluster {d['lca_class_like_alphi_cluster']}); posterior {d['posterior']}")
        for d in lrep["disagreements"]:
            d["id"] = str(ids.iloc[d["index"]])
        lrep["ids"] = [str(x) for x in ids]
        with open(result_folder / "lca_check.json", "w", encoding="utf-8") as fh:
            json.dump({k: v for k, v in lrep.items()}, fh, indent=1, default=float)
        df_transformed["HDBSCAN_hybrid"] = lrep["hybrid_labels"]
        k_h = len(set(lrep["hybrid_labels"]) - {-1})
        flag_results["HDBSCAN_hybrid"] = (flag_few_feature_clusters(
            lrep["hybrid_labels"], symptom_sets, args.flag_delta, args.flag_top, args.flag_max_features), k_h)
        primary_cluster_col = "HDBSCAN_hybrid"
        print(f"  Final clustering exported as optimal: {lrep['adopted']} ({k_h} group(s)); details in lca_check.json")

    # -------------------------------------------------------------------------
    # COHORT-WIDE ARCHETYPAL SIGNATURE (EXECUTED ONCE PER PIPELINE RUN)
    # -------------------------------------------------------------------------
    print("\n==================================================")
    print(f"COHORT-WIDE ARCHETYPAL PHENOTYPIC SIGNATURE (N = {n_cohort})")
    print("==================================================")
    cohort_archetypes = extract_cohort_archetypal_phenotypes(
        df_transformed=df_transformed,
        patient_lookup=patient_lookup,
        tree_lookup=tree_lookup,
        threshold=0.50,
        min_depth_from_root=3
    )

    if cohort_archetypes:
        print(tabulate(cohort_archetypes, headers="keys", tablefmt="pipe"))
    else:
        print("  [Note] No terms reached the ≥50% cohort-wide prevalence threshold.")


    result_dir = Path(result_folder)
    out_csv = result_dir / "pairD_UMAP.csv"
    out_txt = result_dir / "pairD_UMAP.txt" 

    # Efficient File I/O
    df_transformed.to_csv(out_csv, index=False)
    # Replaced memory-heavy to_string with streaming tab-delimited write
    df_transformed.to_csv(out_txt, sep="\t", float_format="%.4f", index=False)

    # Export coordinates payload to JSON for web/Plotly visualizations
    target_json_path = (
        Path(args.output_json) if args.output_json 
        else result_dir / "cluster_visualization.json"
    )

    # Extract HDBSCAN columns, ensuring primary_cluster_col is prioritized if present
    hdbscan_cols = [c for c in df_transformed.columns if c.startswith("HDBSCAN_")]
    if primary_cluster_col in hdbscan_cols:
        hdbscan_cols.remove(primary_cluster_col)
        hdbscan_cols.insert(0, primary_cluster_col)

    export_eligible_clusters_to_json(
        df_transformed, 
        patient_lookup, 
        tree_lookup, 
        hdbscan_cols, 
        str(target_json_path),
        cohort_archetypes=cohort_archetypes
    )

    # Few-feature split flag: log, few_feature_flags.csv, and fields in the cluster JSON
    if any(f for f, _ in flag_results.values()):
        print("\n=== FEW-FEATURE SPLIT CHECK (pairs of clusters distinguished by <= "
              f"{args.flag_max_features} symptom(s) differing by >= {args.flag_delta:.0%}; each cluster vs its closest) ===")
        flag_rows = []
        for col, (fl, k) in flag_results.items():
            for c, v in fl.items():
                tops = "; ".join(f"{get_hpo_label(t, tree_lookup)} {pin:.0%} vs {pout:.0%}" for t, pin, pout in v["top"])
                print(f"  {col} (k = {k}) cluster {c} vs cluster {v['partner']}: {v['n_features']} "
                      f"discriminating symptom(s){'  <-- FLAGGED' if v['flag'] else ''} | {tops}")
                flag_rows.append({"clustering_column": col, "num_clusters": k, "cluster_id": c,
                                  "closest_cluster": v["partner"], "flag_few_features": v["flag"], "n_discriminating_features": v["n_features"],
                                  "top_discriminating": tops})
        pd.DataFrame(flag_rows).to_csv(result_dir / "few_feature_flags.csv", index=False)
        try:
            with open(target_json_path, "rb") as f:
                payload = orjson.loads(f.read())
            for rank_key, res in payload.get("clustering_results", {}).items():
                fl = flag_results.get(res.get("clustering_column"))
                if not fl:
                    continue
                res["any_few_feature_cluster"] = any(v["flag"] for v in fl[0].values())
                for cl in res.get("clusters", []):
                    v = fl[0].get(cl.get("cluster_id"))
                    if v:
                        cl["flag_few_features"] = v["flag"]
                        cl["n_discriminating_features"] = v["n_features"]
                        cl["closest_cluster"] = v["partner"]
                        cl["top_discriminating"] = [{"hpo_code": t, "cluster": pin, "closest_cluster": pout} for t, pin, pout in v["top"]]
            with open(target_json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            print(f"[!] Could not add few-feature flags to {target_json_path}: {e}")

    # Resampling stability: log, stability.csv, and fields in the cluster JSON
    if stability_results:
        print("\n=== RESAMPLING STABILITY (Jaccard recovery; >= 0.75 stable, <= 0.5 not supported) ===")
        stab_rows = []
        for col, (stab, kc, k) in stability_results.items():
            k_txt = ", ".join(f"k={kk}: {vv}" for kk, vv in sorted(kc.items()))
            print(f"  {col} (k = {k}) | k over subsamples: {k_txt}")
            for c, v in stab.items():
                verdict = ("stable" if v["jaccard_mean"] >= 0.75 else
                           "not supported" if v["jaccard_mean"] <= 0.5 else "intermediate")
                print(f"      cluster {c}: mean Jaccard {v['jaccard_mean']:.3f} | dissolved in "
                      f"{v['dissolved']:.0%} of {v['n_resamples']} subsamples | {verdict}")
                stab_rows.append({"clustering_column": col, "num_clusters": k, "cluster_id": c, **v,
                                  "verdict": verdict})
        pd.DataFrame(stab_rows).to_csv(result_dir / "stability.csv", index=False)
        try:
            with open(target_json_path, "rb") as f:
                payload = orjson.loads(f.read())
            for rank_key, res in payload.get("clustering_results", {}).items():
                st = stability_results.get(res.get("clustering_column"))
                if not st:
                    continue
                res["stability_k_counts"] = {str(kk): vv for kk, vv in sorted(st[1].items())}
                for cl in res.get("clusters", []):
                    v = st[0].get(cl.get("cluster_id"))
                    if v:
                        cl["stability_jaccard_mean"] = round(v["jaccard_mean"], 4)
                        cl["stability_dissolved"] = round(v["dissolved"], 4)
            with open(target_json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            print(f"[!] Could not add stability to {target_json_path}: {e}")
        print(f"Stability table: {result_dir / 'stability.csv'}")

    # Run Cluster Coherence & Survival Evaluation
    if not args.skip_evaluation:
        for col in hdbscan_cols:
            evaluate_cluster_phenotypic_coherence(
                df_transformed, 
                patient_lookup, 
                tree_lookup, 
                col, 
                sel_genes=sel_genes,
                digestive_only=digestive_only,
                X=X
            )
            evaluate_cluster_survival(
                df_transformed, 
                patient_lookup, 
                col
            )
            evaluate_cox_ph_model_continuous(
                df_transformed, 
                patient_lookup, 
                col
            )
    else:
        print("\n[Notice] Skipping Cluster Coherence & Survival Evaluation as requested.")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 80}")
    print(f"Pipeline Finished Successfully in {elapsed:.2f} seconds.")
    print(f"Outputs written to directory: {result_dir}")
    print(f" -> Traceability Script: {copied_script_path}")
    print(f" -> Distance Matrix: {cdm_file}")
    print(f" -> {clustering_dims}D UMAP & Clusters CSV: {out_csv}")
    print(f" -> {clustering_dims}D UMAP & Clusters TXT: {out_txt}")
    print(f" -> Cluster Visualization JSON: {target_json_path}")
    print(f" -> Execution Log: {log_file_path}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
