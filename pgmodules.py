#!/usr/bin/env python3.14
import numpy
import json
import math
import os

# 261015: the two values below can be changed for testing through environment
# variables (e.g. PGM_ORGAN_PENALTY=0 alphi.py ...). Defaults = original method.
#   PGM_PERFECT_MATCH : distance given to two identical symptoms (dHPO), default -0.5
#   PGM_ORGAN_PENALTY : added per symptom when an organ system is involved in one patient
#                       and not the other (dPhenome hamm_dist), default 0.5
PERFECT_MATCH = float(os.environ.get("PGM_PERFECT_MATCH", "-0.5"))
ORGAN_PENALTY = float(os.environ.get("PGM_ORGAN_PENALTY", "0.5"))
#   PGM_ORGAN_MODE : how an organ system involved in only one of the two patients is handled
#       "nearest" (default since 261015) : each of those symptoms enters the average like any
#                   other symptom, with its distance to the closest symptom the other patient
#                   has anywhere in the HPO tree (all branches, across organ systems)
#       "penalty" (original method, up to 261015) : + ORGAN_PENALTY per symptom, added to the
#                   final distance
#   Why: with "penalty", a single uncommon symptom that is the only one in its organ system
#   (e.g. inguinal hernia in Marfan syndrome, 16% of patients) adds a flat +0.5 to every
#   distance between patients with and without it. This step made alphi.py report
#   spurious clusters in simulated cohorts with no subgroups (2 clusters in ~2/3 of runs from
#   ~180 patients). With "nearest", 5/5 no-subgroup cohorts gave 1 cluster and 5/5 cohorts
#   with two real subgroups gave 2 clusters (N = 240), and real subgroups separated better
#   (silhouette 0.57 -> 0.60).
ORGAN_MODE = os.environ.get("PGM_ORGAN_MODE", "nearest")
#   PGM_DISTANCE : distance between two patients returned by dPhenome() (261016, baseline
#   comparisons; alphi.py is unchanged and calls dPhenome() as before)
#       "dphenome" (default) : the ontology-aware phenome distance below
#       "jaccard"            : 1 - |A & B| / |A | B| on the sets of reported HPO codes
#       "hamming"            : |A ^ B|, the number of codes reported in one patient only
#       "resnik"             : 1 - Resnik best-match average (BMA) similarity, the standard
#                              ontology-based phenotype similarity (as in Phenomizer):
#                              sim(a, b) = IC of the most informative common ancestor,
#                              BMA = mean of (average best match of A's terms in B,
#                                             average best match of B's terms in A)
#                              Information content (IC), set by PGM_RESNIK_IC:
#         "annotation" (default): from disease annotations (as in Phenomizer):
#              IC(t) = -log(n(t) / N), n(t) = diseases annotated with t or a descendant
#              of t, N = annotated diseases; read from the HPO annotation file
#              phenotype.hpoa (PGM_HPOA, default: next to this file); 'NOT' annotations
#              and non-phenotype aspects are left out; terms never used count as n = 1.
#              PGM_HPOA_DB restricts the diseases (e.g. "ORPHA" or "OMIM,ORPHA").
#              The table is cached in <phenotype.hpoa>.ic.json (rebuilt when the
#              annotation file or the tree changes).
#         "intrinsic": from the HPO structure (Seco et al. 2004):
#              IC(t) = -log((descendants(t) + 1) / number of terms)
#                              IC is divided by its maximum, so 0 <= IC <= 1 and the distance
#                              is between 0 and 1. Ancestors come from tree_hashtable: needs
#                              the COMPLETE hp_trees.json (build_hpo_lookup_files.py).
#   jaccard and hamming compare exact codes only (no HPO tree, no organ systems).
DISTANCE_MODE = os.environ.get("PGM_DISTANCE", "dphenome")
RESNIK_IC = os.environ.get("PGM_RESNIK_IC", "annotation")
HPOA_PATH = os.environ.get("PGM_HPOA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "phenotype.hpoa"))
HPOA_DB = os.environ.get("PGM_HPOA_DB", "")          # empty = all disease databases
DISTANCE_MODES = ("dphenome", "jaccard", "hamming", "resnik")

# 261111: optional term weights (alphi.py --select-terms, see term_selection.py).
# None (default) = every term counts equally, as before. Otherwise {code: weight}: OB is the
# weighted mean of the best-match distances (each weighted by the term being matched),
# Jaccard the weighted Jaccard index and Resnik the weighted best-match average; codes not
# in the dictionary have weight 1.
TERM_WEIGHTS = None


def set_term_weights(weights):
    """Sets the term weights used by dPhenome() (None = unweighted)."""
    global TERM_WEIGHTS
    TERM_WEIGHTS = dict(weights) if weights else None


def _w(code):
    if TERM_WEIGHTS is None:
        return 1.0
    return TERM_WEIGHTS.get(str(code).replace("HP:", "").zfill(7), 1.0)
if DISTANCE_MODE not in DISTANCE_MODES:
    raise ValueError(f"PGM_DISTANCE must be one of {DISTANCE_MODES}, not {DISTANCE_MODE!r}")

# ----------------------------------
# README
# -----------
# Started 251219. Based on build_symptom_tables.py
# Objective 1 -- to calculate distances between two symptoms in terms of HPO nodes
#                how many HPO nodes are there between symptom A and symptom B
# Continued 260804. Saved as pgmodules.py
# 260416 NOTES --> Some organ systems are more important than others 
#                  (e.g. with and without facial features considered to have same disease but in HPO space they would be considered different)
#                  Should I add hamming distances for specific "characteristic" symptoms?
#                  For organ-system hamming distances, should I include a threshold minimum number of hits?
#                  Measure distances between HPO-definitions of diseases
#                  The hamming distance penalty should be inversely proportional to the number of symptoms reported
#                  Get weightings for different systems from phenomeprint analysis of all patients (weights should be proportional to overall prevalence of system)
#                  I need to check that the clusters are associated with distance and not with doi or other potential confounders (number of symptoms reported?)
#                  The hamm distance penalty should be related to the average distance per system for the systems with symptoms present?
#                      This is not really feasible because it depends on the average over all pairwise distances (whereas it is counted at the level of single pairwise distances)
#                      Instead, set the hamm penalty equal to twice the value for a perfect match (i.e. 2 * 5 = 10)
#                  And what if I make perfect matches give a distance of -1 rather than 0 (and then take the total distance as max(0, d)
#
#260513: TO DO, write out distances in different organ systems separately
#
#
#260622: Gemini-optimized code (mostly via lookups)


def intersect(a, b):
    return sorted(list(set(a) & set(b)))


def check_opposite_directional_terms(codeA, codeB, id_to_term_map):
    """
    Returns True if codeA and codeB have identical term descriptions,
    except one starts with ('elevated', 'increased') and the other
    starts with ('decreased', 'reduced').
    """
    if not id_to_term_map or not codeA or not codeB:
        return False

    # Normalize HPO ID format (e.g., '0030948' -> 'HP:0030948')
    idA = codeA if str(codeA).startswith("HP:") else f"HP:{codeA}"
    idB = codeB if str(codeB).startswith("HP:") else f"HP:{codeB}"

    termA = id_to_term_map.get(idA)
    termB = id_to_term_map.get(idB)

    if not termA or not termB:
        return False

    inc_prefixes = ("elevated ", "increased ")
    dec_prefixes = ("decreased ", "reduced ")

    termA_lower = termA.lower()
    termB_lower = termB.lower()

    def extract_direction_and_stem(term):
        for prefix in inc_prefixes:
            if term.startswith(prefix):
                return "INC", term[len(prefix):].strip()
        for prefix in dec_prefixes:
            if term.startswith(prefix):
                return "DEC", term[len(prefix):].strip()
        return None, term

    dirA, stemA = extract_direction_and_stem(termA_lower)
    dirB, stemB = extract_direction_and_stem(termB_lower)

    is_opposite = (dirA == "INC" and dirB == "DEC") or (dirA == "DEC" and dirB == "INC")
    return is_opposite and (stemA == stemB)


def dHPO(*args, id_to_term_map=None):
    # --- 1. Fast Argument Parsing & Type Checking ---
    num_args = len(args)
    if not 2 <= num_args <= 4:
        raise TypeError(f'dHPO() takes between 2 and 4 arguments ({num_args} given)')

    branchA, branchB = args[0], args[1]

    if not isinstance(branchA, list) or not isinstance(branchB, list):
        raise TypeError('dHPO takes two lists as arguments')

    different_organ_penalty = 0
    if num_args >= 3 and args[2] is not None:
        different_organ_penalty = int(args[2])
        if not 0 <= different_organ_penalty < 999:
            raise TypeError(f'dHPO() allows penalties between 0 and 999 ({different_organ_penalty} given)')

    if num_args == 4 and id_to_term_map is None:
        id_to_term_map = args[3]

    # --- 2. Compute Intersection ---
    setB = set(branchB)
    common = [code for code in branchA if code in setB]
    
    if not common:
        return 999  # Fallback if zero overlap

    # --- 3. Correct Lowest Common Ancestor (LCA) Distance ---
    d_AB = min(branchA.index(code) + branchB.index(code) for code in common)

    # --- 4. Root Ancestor Check ---
    if set(common).issubset({'0000001', '0000118', '-------'}):
        d_AB += different_organ_penalty

    # --- 5. Directional Term Opposition Penalty (+5) ---
    if id_to_term_map and branchA and branchB:
        if check_opposite_directional_terms(branchA[0], branchB[0], id_to_term_map):
            d_AB += 5

    # Perfect match adjustment
    if d_AB == 0:
        return PERFECT_MATCH
        
    return d_AB


def dPhenome_original(patientA_dict, patientB_dict, tree_hashtable, digestive_only=True, id_to_term_map=None):
    """
    Calculates the phenome distance between two patients.
    Original implementation, kept unchanged for validation; dPhenome() below gives
    the same result faster.
    """
    if not isinstance(patientA_dict, dict) or not isinstance(patientB_dict, dict):
        raise TypeError('dPhenome takes two dictionaries as its first arguments')

    organ_systems = [
        {'name': 'musculokeletal', 'code': '0033127', 'weight': .238},
        {'name': 'limbs', 'code': '0040064', 'weight': .0353},
        {'name': 'nerves', 'code': '0000707', 'weight': .135},
        {'name': 'metabolism', 'code': '0001939', 'weight': 2.13}, 
        {'name': 'genitourinary', 'code': '0000119', 'weight': .85},
        {'name': 'cellular', 'code': '0025354', 'weight': .003},
        {'name': 'immune', 'code': '0002715', 'weight': .356},
        {'name': 'eye', 'code': '0000478', 'weight': .056},
        {'name': 'integument', 'code': '0001574', 'weight': .418},
        {'name': 'voice', 'code': '0001608', 'weight': 0.},
        {'name': 'digestive', 'code': '0025031', 'weight': 10.}, 
        {'name': 'ear', 'code': '0000598', 'weight': 0.},
        {'name': 'cardiovascular', 'code': '0001626', 'weight': .019},
        {'name': 'prenatal', 'code': '0001197', 'weight': .185},
        {'name': 'neoplasm', 'code': '0002664', 'weight': 0.},
        {'name': 'head/neck', 'code': '0000152', 'weight': .038},
        {'name': 'endocrine', 'code': '0000818', 'weight': .05},
        {'name': 'constitutional', 'code': '0025142', 'weight': 0.},
        {'name': 'growth', 'code': '0001507', 'weight': .797},
        {'name': 'thoracic', 'code': '0045027', 'weight': 0.},
        {'name': 'respiratory', 'code': '0002086', 'weight': .079},
        {'name': 'blood', 'code': '0001871', 'weight': 0.141}, 
        {'name': 'breast', 'code': '0000769', 'weight': 0}
    ]
    
    d_penalty = 0 
    pairwise = []
    all_distances = []

    patientA_codes = set(extract_symptom_codes(patientA_dict.get('symptoms', [])))
    patientB_codes = set(extract_symptom_codes(patientB_dict.get('symptoms', [])))

    for organ_dict in organ_systems:
        organ_name = organ_dict['name']    
        organ_code = organ_dict['code']    
        organ_weight = 1

        if digestive_only and organ_name != "digestive": 
            continue

        # Extract branches relevant to this organ system for Patient A
        organ_branchesA = {}
        for code in patientA_codes:
            entry = tree_hashtable.get(code)
            if entry:
                valid_parents = [p for p in entry.get("parent_branches", []) if organ_code in p]
                if valid_parents:
                    organ_branchesA[code] = valid_parents

        # Extract branches relevant to this organ system for Patient B
        organ_branchesB = {}
        for code in patientB_codes:
            entry = tree_hashtable.get(code)
            if entry:
                valid_parents = [p for p in entry.get("parent_branches", []) if organ_code in p]
                if valid_parents:
                    organ_branchesB[code] = valid_parents

        sympcountA = len(organ_branchesA)
        sympcountB = len(organ_branchesB)

        symptom_distances = []

        # Direction 1: Patient A -> Patient B
        for codeA, parentsA in organ_branchesA.items():
            min_dist = float('inf')
            for codeB, parentsB in organ_branchesB.items():
                for pA in parentsA:
                    for pB in parentsB:
                        d_val = dHPO(pA, pB, d_penalty, id_to_term_map) * organ_weight
                        if d_val < min_dist:
                            min_dist = d_val
            if min_dist != float('inf'):
                symptom_distances.append(min_dist)

        # Direction 2: Patient B -> Patient A
        for codeB, parentsB in organ_branchesB.items():
            min_dist = float('inf')
            for codeA, parentsA in organ_branchesA.items():
                for pB in parentsB:
                    for pA in parentsA:
                        d_val = dHPO(pB, pA, d_penalty, id_to_term_map) * organ_weight
                        if d_val < min_dist:
                            min_dist = d_val
            if min_dist != float('inf'):
                symptom_distances.append(min_dist)

        pairwise_el = { 
            'distances': symptom_distances, 
            'mean_distance': numpy.mean(symptom_distances) if symptom_distances else None,
            'organ_name': organ_name,
            'organ_code': organ_code,
            'sympcountA': sympcountA,
            'sympcountB': sympcountB
        }
        pairwise.append(pairwise_el)

    # --- Aggregations and Penalties ---
    hamm_dist = 0
    for item in pairwise:
        organ_dists = item['distances']
        sympcountA = item['sympcountA']
        sympcountB = item['sympcountB']

        if organ_dists:
            all_distances.extend(organ_dists)
        elif sympcountA != sympcountB:
            hamm_dist += abs(sympcountA - sympcountB) * 0.5
            
    if all_distances and TERM_WEIGHTS is not None and sum(all_weights) > 0:
        d = sum(w * x for w, x in zip(all_weights, all_distances)) / sum(all_weights) + hamm_dist
        d = max(0, d)
    elif all_distances:
        d = sum(all_distances) / len(all_distances) + hamm_dist
        d = max(0, d)
    else:
        d = hamm_dist

    return d


# ----------------------------------------------------------------------------
# 261015: Cached implementation of dPhenome (same result as dPhenome_original).
# Two things are recomputed many times by the original when distances are
# calculated for every pair of patients:
#   1. a symptom's HPO branches within an organ system (depends only on the
#      symptom and the organ, not on the pair);
#   2. the smallest dHPO between two symptoms over all their branches (depends
#      only on the two symptoms and the organ).
# Both are computed once and stored. The stores are cleared whenever a different
# HPO tree or descriptor map is passed in, so results never mix trees.
# ----------------------------------------------------------------------------

ORGAN_SYSTEMS = (
    ('musculokeletal', '0033127'), ('limbs', '0040064'), ('nerves', '0000707'),
    ('metabolism', '0001939'), ('genitourinary', '0000119'), ('cellular', '0025354'),
    ('immune', '0002715'), ('eye', '0000478'), ('integument', '0001574'),
    ('voice', '0001608'), ('digestive', '0025031'), ('ear', '0000598'),
    ('cardiovascular', '0001626'), ('prenatal', '0001197'), ('neoplasm', '0002664'),
    ('head/neck', '0000152'), ('endocrine', '0000818'), ('constitutional', '0025142'),
    ('growth', '0001507'), ('thoracic', '0045027'), ('respiratory', '0002086'),
    ('blood', '0001871'), ('breast', '0000769'),
)

_cache_tree = None
_cache_map = None
_cache_perfect = None
_branch_cache = {}      # (code, organ_code) -> list of branches within that organ (or [])
_pair_cache = {}        # (organ_code, codeA, codeB) -> min dHPO over their branches
_profile_cache = {}     # frozenset of a patient's codes -> {organ_code: {code: branches}}


def _check_cache(tree_hashtable, id_to_term_map):
    """Clears the stores if a different tree, descriptor map or perfect-match value is used."""
    global _cache_tree, _cache_map, _cache_perfect
    if tree_hashtable is not _cache_tree or id_to_term_map is not _cache_map \
            or PERFECT_MATCH != _cache_perfect:
        _branch_cache.clear()
        _pair_cache.clear()
        _profile_cache.clear()
        _cache_tree, _cache_map, _cache_perfect = tree_hashtable, id_to_term_map, PERFECT_MATCH


def _organ_branches(code, organ_code, tree_hashtable):
    key = (code, organ_code)
    hit = _branch_cache.get(key)
    if hit is None:
        entry = tree_hashtable.get(code)
        hit = [p for p in entry.get("parent_branches", []) if organ_code in p] if entry else []
        _branch_cache[key] = hit
    return hit


def _all_branches(code, tree_hashtable):
    """All branches of a symptom, regardless of organ system (cached)."""
    key = (code, "*")
    hit = _branch_cache.get(key)
    if hit is None:
        entry = tree_hashtable.get(code)
        hit = list(entry.get("parent_branches", [])) if entry else []
        _branch_cache[key] = hit
    return hit


def _nearest_anywhere(code, other_codes, tree_hashtable, id_to_term_map):
    """Distance from a symptom to the closest of the other patient's symptoms, over the whole tree."""
    parents = _all_branches(code, tree_hashtable)
    best = float('inf')
    for other in other_codes:
        other_parents = _all_branches(other, tree_hashtable)
        if parents and other_parents:
            d_val = _min_pair_distance(code, other, parents, other_parents, "*", id_to_term_map)
            if d_val < best:
                best = d_val
    return best


def _min_pair_distance(codeA, codeB, parentsA, parentsB, organ_code, id_to_term_map):
    # dHPO is symmetric in its two branches, so one entry serves both directions.
    key = (organ_code, codeA, codeB) if codeA <= codeB else (organ_code, codeB, codeA)
    hit = _pair_cache.get(key)
    if hit is None:
        hit = float('inf')
        for pA in parentsA:
            for pB in parentsB:
                d_val = dHPO(pA, pB, 0, id_to_term_map)
                if d_val < hit:
                    hit = d_val
        _pair_cache[key] = hit
    return hit


def _patient_profile(codes, tree_hashtable):
    """
    {organ_code: {code: branches}} for one patient, in the same symptom order as the
    original (iteration order of the patient's code set). Computed once per distinct
    set of codes.
    """
    key = frozenset(codes)
    prof = _profile_cache.get(key)
    if prof is None:
        prof = {}
        for _, organ_code in ORGAN_SYSTEMS:
            branches = {}
            for code in codes:
                b = _organ_branches(code, organ_code, tree_hashtable)
                if b:
                    branches[code] = b
            prof[organ_code] = branches
        _profile_cache[key] = prof
    return prof


def set_distance(patientA_dict, patientB_dict, mode):
    """Baseline distances on the sets of reported HPO codes (see PGM_DISTANCE above)."""
    A = {str(c).replace("HP:", "").zfill(7) for c in extract_symptom_codes(patientA_dict.get('symptoms', []))}
    B = {str(c).replace("HP:", "").zfill(7) for c in extract_symptom_codes(patientB_dict.get('symptoms', []))}
    if mode == "hamming":
        return float(len(A ^ B))
    if mode == "jaccard":
        union = A | B
        if TERM_WEIGHTS is not None and union and sum(_w(c) for c in union) > 0:
            return 1.0 - sum(_w(c) for c in A & B) / sum(_w(c) for c in union)
        return 1.0 - len(A & B) / len(union) if union else 0.0
    raise ValueError(f"unknown distance mode {mode!r}")


_resnik_tree = None
_resnik_key = None
_resnik_anc = {}        # code -> frozenset of its ancestors (itself included)
_resnik_ic = {}         # code -> information content, scaled to 0..1
_resnik_sim = {}        # (code, code) -> Resnik similarity


def _hpoa_counts(current, hpoa_path, dbs):
    """{code: number of annotated diseases using the code or a descendant}, and N."""
    import csv as _csv
    diseases = {}
    with open(hpoa_path, encoding="utf-8") as f:
        rows = (line for line in f if not line.startswith("#"))
        reader = _csv.reader(rows, delimiter="\t")
        header = next(reader)
        col = {h: i for i, h in enumerate(header)}
        for r in reader:
            if len(r) <= max(col["database_id"], col["hpo_id"], col["aspect"], col["qualifier"]):
                continue
            if r[col["aspect"]] != "P" or r[col["qualifier"]].strip().upper() == "NOT":
                continue
            db = r[col["database_id"]].split(":")[0]
            if dbs and db not in dbs:
                continue
            code = r[col["hpo_id"]].replace("HP:", "").zfill(7)
            diseases.setdefault(r[col["database_id"]], set()).add(code)
    counts = dict.fromkeys(current, 0)
    for codes in diseases.values():
        closure = set()
        for c in codes:
            closure |= _resnik_anc.get(c, {c})
        for a in closure:
            if a in counts:
                counts[a] += 1
    return counts, len(diseases)


def resnik_ic_description():
    """Short description of the IC in use, or an error message if it cannot be built."""
    if RESNIK_IC == "intrinsic":
        return "intrinsic IC (HPO structure)"
    if RESNIK_IC != "annotation":
        raise ValueError(f"PGM_RESNIK_IC must be 'annotation' or 'intrinsic', not {RESNIK_IC!r}")
    if not os.path.exists(HPOA_PATH):
        raise FileNotFoundError(f"Resnik with annotation-based IC needs the HPO annotation file "
                                f"phenotype.hpoa at {HPOA_PATH} (or set PGM_HPOA, or PGM_RESNIK_IC=intrinsic).")
    return f"annotation-based IC ({HPOA_PATH}{', ' + HPOA_DB if HPOA_DB else ''})"


def _resnik_tables(tree_hashtable):
    """Ancestor sets and IC of every term (see PGM_RESNIK_IC above)."""
    global _resnik_tree, _resnik_key
    key = (RESNIK_IC, HPOA_PATH, HPOA_DB)
    if _resnik_tree is tree_hashtable and _resnik_key == key:
        return
    _resnik_anc.clear(); _resnik_ic.clear(); _resnik_sim.clear()
    for code, entry in tree_hashtable.items():
        anc = {str(x).zfill(7) for route in entry.get("parent_branches", []) for x in route}
        anc.discard("-------")
        anc.add(code)
        _resnik_anc[code] = frozenset(anc)
    # retired-ID aliases repeat a current term: counted once, through the current term
    current = [c for c, e in tree_hashtable.items() if not e.get("replaced_by")]
    n_terms = len(current)
    if n_terms < 15000:
        print(f"[pgmodules] WARNING: only {n_terms} terms in the HPO tree; Resnik IC needs the complete "
              f"hp_trees.json (build_hpo_lookup_files.py).", flush=True)

    if RESNIK_IC == "intrinsic":
        n_desc = dict.fromkeys(current, 0)
        for c in current:
            for a in _resnik_anc[c]:
                if a != c and a in n_desc:
                    n_desc[a] += 1
        raw = {c: -math.log((n_desc[c] + 1) / n_terms) for c in current}
        ic_max = math.log(n_terms)
    else:
        resnik_ic_description()                                  # raises if the file is missing
        st = os.stat(HPOA_PATH)
        stamp = [st.st_size, int(st.st_mtime), n_terms, HPOA_DB]
        cache = HPOA_PATH + (f".{HPOA_DB.replace(',', '_')}" if HPOA_DB else "") + ".ic.json"
        raw = None
        try:
            with open(cache, encoding="utf-8") as f:
                d = json.load(f)
            if d.get("stamp") == stamp:
                raw, ic_max = d["ic"], d["ic_max"]
        except (OSError, ValueError, KeyError):
            pass
        if raw is None:
            dbs = {x.strip() for x in HPOA_DB.split(",") if x.strip()}
            counts, n_dis = _hpoa_counts(current, HPOA_PATH, dbs)
            if n_dis == 0:
                raise ValueError(f"No usable annotations in {HPOA_PATH} (PGM_HPOA_DB={HPOA_DB!r}).")
            raw = {c: -math.log(max(counts[c], 1) / n_dis) for c in current}
            ic_max = math.log(n_dis)
            try:
                tmp = cache + f".{os.getpid()}.tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump({"stamp": stamp, "ic_max": ic_max, "n_diseases": n_dis, "ic": raw}, f)
                os.replace(tmp, cache)
            except OSError:
                pass
    for c in current:
        _resnik_ic[c] = raw[c] / ic_max if ic_max > 0 else 0.0
    for code, entry in tree_hashtable.items():          # aliases take their term's IC
        if code not in _resnik_ic:
            rep = str(entry.get("replaced_by", "")).replace("HP:", "").zfill(7)
            _resnik_ic[code] = _resnik_ic.get(rep, 0.0)
    _resnik_tree, _resnik_key = tree_hashtable, key


def _resnik_pair(a, b):
    key = (a, b) if a <= b else (b, a)
    hit = _resnik_sim.get(key)
    if hit is None:
        common = _resnik_anc[a] & _resnik_anc[b]
        hit = max((_resnik_ic.get(c, 0.0) for c in common), default=0.0)
        _resnik_sim[key] = hit
    return hit


def resnik_distance(patientA_dict, patientB_dict, tree_hashtable):
    """1 - Resnik best-match average similarity (see PGM_DISTANCE above)."""
    _resnik_tables(tree_hashtable)
    A = [c for c in {str(x).replace("HP:", "").zfill(7)
                     for x in extract_symptom_codes(patientA_dict.get('symptoms', []))} if c in _resnik_anc]
    B = [c for c in {str(x).replace("HP:", "").zfill(7)
                     for x in extract_symptom_codes(patientB_dict.get('symptoms', []))} if c in _resnik_anc]
    if not A or not B:
        return 1.0
    if TERM_WEIGHTS is not None and sum(_w(a) for a in A) > 0 and sum(_w(b) for b in B) > 0:
        a_to_b = sum(_w(a) * max(_resnik_pair(a, b) for b in B) for a in A) / sum(_w(a) for a in A)
        b_to_a = sum(_w(b) * max(_resnik_pair(a, b) for a in A) for b in B) / sum(_w(b) for b in B)
        return 1.0 - (a_to_b + b_to_a) / 2
    a_to_b = sum(max(_resnik_pair(a, b) for b in B) for a in A) / len(A)
    b_to_a = sum(max(_resnik_pair(a, b) for a in A) for b in B) / len(B)
    return 1.0 - (a_to_b + b_to_a) / 2


def dPhenome(patientA_dict, patientB_dict, tree_hashtable, digestive_only=True, id_to_term_map=None):
    """
    Calculates the phenome distance between two patients (cached implementation).
    With PGM_DISTANCE = "jaccard", "hamming" or "resnik", returns that distance instead.
    With ORGAN_MODE = "penalty" it gives exactly the same result as dPhenome_original;
    with the default ORGAN_MODE = "nearest", organ systems involved in only one patient
    are handled as described at the top of this file.
    """
    if not isinstance(patientA_dict, dict) or not isinstance(patientB_dict, dict):
        raise TypeError('dPhenome takes two dictionaries as its first arguments')

    if DISTANCE_MODE == "resnik":
        return resnik_distance(patientA_dict, patientB_dict, tree_hashtable)
    if DISTANCE_MODE != "dphenome":
        return set_distance(patientA_dict, patientB_dict, DISTANCE_MODE)

    _check_cache(tree_hashtable, id_to_term_map)

    codesA = set(extract_symptom_codes(patientA_dict.get('symptoms', [])))
    codesB = set(extract_symptom_codes(patientB_dict.get('symptoms', [])))

    profA = _patient_profile(codesA, tree_hashtable)
    profB = _patient_profile(codesB, tree_hashtable)

    all_distances = []
    all_weights = []
    hamm_dist = 0

    for organ_name, organ_code in ORGAN_SYSTEMS:
        if digestive_only and organ_name != "digestive":
            continue

        branchesA = profA[organ_code]
        branchesB = profB[organ_code]

        if branchesA and branchesB:
            # Direction 1: each symptom of A to its closest symptom of B
            for codeA, parentsA in branchesA.items():
                best = float('inf')
                for codeB, parentsB in branchesB.items():
                    d_val = _min_pair_distance(codeA, codeB, parentsA, parentsB, organ_code, id_to_term_map)
                    if d_val < best:
                        best = d_val
                all_distances.append(best)
                all_weights.append(_w(codeA))
            # Direction 2: each symptom of B to its closest symptom of A
            for codeB, parentsB in branchesB.items():
                best = float('inf')
                for codeA, parentsA in branchesA.items():
                    d_val = _min_pair_distance(codeA, codeB, parentsA, parentsB, organ_code, id_to_term_map)
                    if d_val < best:
                        best = d_val
                all_distances.append(best)
                all_weights.append(_w(codeB))
        elif len(branchesA) != len(branchesB):
            # One patient has symptoms in this organ system and the other has none
            if ORGAN_MODE == "nearest":
                lone, other_codes = (branchesA, codesB) if branchesA else (branchesB, codesA)
                for code in lone:
                    d_val = _nearest_anywhere(code, other_codes, tree_hashtable, id_to_term_map)
                    if d_val != float('inf'):
                        all_distances.append(d_val)
                        all_weights.append(_w(code))
                    else:   # the other patient has no symptom in the tree at all
                        hamm_dist += ORGAN_PENALTY
            else:
                hamm_dist += abs(len(branchesA) - len(branchesB)) * ORGAN_PENALTY

    if all_distances and TERM_WEIGHTS is not None and sum(all_weights) > 0:
        d = sum(w * x for w, x in zip(all_weights, all_distances)) / sum(all_weights) + hamm_dist
        d = max(0, d)
    elif all_distances:
        d = sum(all_distances) / len(all_distances) + hamm_dist
        d = max(0, d)
    else:
        d = hamm_dist

    return d


def row_col_from_condensed_index(d, index):
    """
    Converts a condensed distance matrix index back to row (i) and col (j) indices.
    """
    b = 1 - (2 * d) 
    disc = b ** 2 - 8 * index
    i = (-b - math.isqrt(disc)) // 2
    j = index + i * (b + i + 2) // 2 + 1
    return int(i), int(j)


def extract_symptom_codes(symptoms_list):
    if not symptoms_list:
        return []
    
    first_element = symptoms_list[0]
    
    if isinstance(first_element, (list, tuple)):
        return [item[0] for item in symptoms_list]
    elif isinstance(first_element, str):
        return symptoms_list
    else:
        raise ValueError(f"Unexpected data format inside symptoms: {type(first_element)}")














