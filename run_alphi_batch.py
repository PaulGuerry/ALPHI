#!/usr/bin/env python3.14
import os
import sys
import subprocess
import orjson
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch executor and JSON concatenator for alphi.py"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="gene_sets.txt",
        help="Text file containing gene combinations (one combination per line)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="combined_cluster_visualizations.json",
        help="Path for the final merged JSON output file"
    )
    parser.add_argument(
        "--extra-args",
        type=str,
        default="",
        help="Additional CLI flags to pass through to alphi.py (e.g. '--all')"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        sys.exit(1)

    # Read gene sets from text file (skip empty lines and comments)
    with open(args.input_file, "r") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    if not lines:
        print("No valid gene sets found in input file.")
        sys.exit(1)

    combined_results = {}

    for line in lines:
        genes = line.split()
        # Build a unique key name for the JSON root (e.g. "MYO5B_STX3")
        gene_key = "_".join(sorted(genes))
        temp_json_path = f"temp_viz_{gene_key}.json"

        # Construct alphi.py command
        cmd = [
            sys.executable, "alphi.py",
            "--gene"
        ] + genes + [
            "--output-json", temp_json_path
        ]

        # Pass through extra CLI flags if provided
        if args.extra_args:
            cmd.extend(args.extra_args.split())

        print(f"\n==================================================")
        print(f" Running alphi.py for gene set: {gene_key}")
        print(f" Command: {' '.join(cmd)}")
        print(f"==================================================")

        # Run pipeline
        try:
            res = subprocess.run(cmd, check=True)
            
            # Read and append generated JSON if successful
            if os.path.exists(temp_json_path):
                with open(temp_json_path, "rb") as jf:
                    data = orjson.loads(jf.read())
                
                # Attach to combined dictionary under the gene combination key
                combined_results[gene_key] = data
                
                # Cleanup temporary file
                os.remove(temp_json_path)
            else:
                print(f"[Warning] Expected JSON output '{temp_json_path}' was not found.")

        except subprocess.CalledProcessError as e:
            print(f"[Error] Execution failed for gene set '{gene_key}' with exit code {e.returncode}.")
            continue

    # Export merged master JSON
    if combined_results:
        with open(args.output_file, "wb") as f:
            f.write(orjson.dumps(combined_results, option=orjson.OPT_INDENT_2))

        print(f"\n==================================================")
        print(f" Batch execution complete.")
        print(f" Combined output successfully written to: {args.output_file}")
        print(f" Top-level keys included: {list(combined_results.keys())}")
        print(f"==================================================")
    else:
        print("\nNo results were collected to merge.")

if __name__ == "__main__":
    main()
