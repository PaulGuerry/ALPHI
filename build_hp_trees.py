#!/usr/bin/env python3.14
"""
build_hp_trees.py

Generates pre-calculated tree structures (hp_trees.json) for ALL terms in the 
Human Phenotype Ontology (HPO) graph. The output JSON file maps each 7-digit 
HPO code to its term description, upstream DAG branches to the root ('0000001'), 
and branch lengths for downstream phenotypic distance analysis.
"""

import argparse
import json
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build full HPO tree structures for all ontology terms."
    )
    parser.add_argument(
        "--hpo-json",
        default="260623_hp_pg.json",
        help="Path to the HPO graph JSON file (default: 260623_hp_pg.json)",
    )
    parser.add_argument(
        "--output-json",
        default="hp_trees.json",
        help="Path to save the generated HPO tree structure JSON (default: hp_trees.json)",
    )
    return parser.parse_args()


def extract_hpo_code(uri_or_id: str) -> str:
    """Extracts a standard 7-digit HPO code from a URI or raw identifier string."""
    if not uri_or_id:
        return ""
    if "HP_" in uri_or_id:
        return uri_or_id.rsplit("HP_", 1)[-1]
    return uri_or_id[-7:]


def find_all_paths(current_code: str, edge_map: dict, memo: dict = None, visited: set = None) -> list:
    """
    Recursively finds all upstream path traversals from current_code to the HPO root ('0000001').
    Uses memoization to optimize traversal performance across the full ontology DAG.
    """
    if memo is None:
        memo = {}
    if visited is None:
        visited = set()

    if current_code in memo:
        return memo[current_code]

    if current_code in visited:
        return [[current_code]]
    visited.add(current_code)

    # Base cases: root node reached or term has no registered parents
    if current_code == "0000001":
        res = [["0000001"]]
        memo[current_code] = res
        return res
    if current_code not in edge_map:
        res = [[current_code]]
        memo[current_code] = res
        return res

    paths = []
    for parent in edge_map[current_code]:
        if parent not in visited:
            parent_paths = find_all_paths(parent, edge_map, memo, visited.copy())
            for path in parent_paths:
                paths.append([current_code] + path)
        else:
            paths.append([current_code, parent])

    memo[current_code] = paths
    return paths


def main():
    args = parse_args()
    hpo_json_path = Path(args.hpo_json)
    output_json_path = Path(args.output_json)

    if not hpo_json_path.exists():
        print(f"Error: Input HPO JSON file '{hpo_json_path}' not found.", file=sys.stderr)
        sys.exit(1)

    # 1. Load HPO Graph
    print(f"Loading HPO dataset from {hpo_json_path}...")
    with open(hpo_json_path, "r", encoding="utf-8") as f:
        hpo_data = json.load(f)

    hpo_graph = hpo_data.get("graphs", [{}])[0]
    hpo_nodes_list = hpo_graph.get("nodes", [])
    hpo_edges_list = hpo_graph.get("edges", [])

    print(f"Successfully loaded {len(hpo_nodes_list)} nodes and {len(hpo_edges_list)} edges.")

    # 2. Build fast lookup maps (O(1) lookup)
    print("Building node label and edge relationship indices...")
    node_lbl_map = {}
    for item in hpo_nodes_list:
        node_id = item.get("id", "")
        code = extract_hpo_code(node_id)
        if len(code) == 7 and code.isdigit():
            node_lbl_map[code] = item.get("lbl", "")

    edge_map = {}
    for item in hpo_edges_list:
        sub_code = extract_hpo_code(item.get("sub", ""))
        obj_code = extract_hpo_code(item.get("obj", ""))
        if len(sub_code) == 7 and sub_code.isdigit() and len(obj_code) == 7 and obj_code.isdigit():
            edge_map.setdefault(sub_code, []).append(obj_code)

    # 3. Process all HPO terms
    print(f"Traversing ontology trees for {len(node_lbl_map)} unique HPO terms...")
    HPO_code_library = []
    total_branches_created = 0
    memo = {}

    for child_code in sorted(node_lbl_map.keys()):
        lbl = node_lbl_map.get(child_code, "")
        raw_paths = find_all_paths(child_code, edge_map, memo=memo)

        # Pad paths with "-------" to equalize array length within the record
        max_len = max(len(p) for p in raw_paths) if raw_paths else 0
        parent_branches = []
        branch_lengths = []

        for path in raw_paths:
            length = path.index("0000001") if "0000001" in path else len(path) - 1
            branch_lengths.append(length)

            padded_path = path + ["-------"] * (max_len - len(path))
            parent_branches.append(padded_path)

        HPO_code_library.append({
            "code": child_code,
            "description": lbl,
            "parent_branches": parent_branches,
            "branch_lengths": branch_lengths
        })

        total_branches_created += len(parent_branches)

    print(f"Analyzed {len(HPO_code_library):5d} unique HPO terms.")
    print(f"Generated {total_branches_created:5d} total HPO tree branches.")

    # 4. Export Output
    print(f"Writing full HPO tree structure to {output_json_path}...")
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as outfile:
        json.dump(HPO_code_library, outfile, indent=4, sort_keys=True)

    print("Done!")


if __name__ == "__main__":
    main() 
