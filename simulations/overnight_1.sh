#!/usr/bin/env bash
# Overnight run 1: Simulation 2, full grid with the latent class check, compositions C1-C3.
# Run in tmux:  bash ~/pairD_eval/scripts/overnight_1.sh
# A grid that fails does not stop the next one; each start and end is written to overnight_1.log.
set -u
E=~/pairD_eval
LOG=$E/overnight_1.log
echo "$(date '+%F %T') overnight_1 started" >> "$LOG"
for C in C1 C2 C3; do
  (
    mkdir -p "$E/test1/lca/$C" && cd "$E/test1/lca/$C" || exit 1
    echo "$(date '+%F %T') Simulation 2 $C started" >> "$LOG"
    python3.14 "$E/scripts/run_test1_grid.py" -t "$E/test1/inputs/tables/test1_table_$C.csv" \
        --hpo-terms "$E/inputs/hpo_terms.json" --distance dphenome --paird-args="--lca" \
        -o lca_results.csv --raw lca_iterations.csv 2>&1 | tee lca.log
    echo "$(date '+%F %T') Simulation 2 $C ended (exit ${PIPESTATUS[0]})" >> "$LOG"
  )
done
echo "$(date '+%F %T') overnight_1 finished" >> "$LOG"
