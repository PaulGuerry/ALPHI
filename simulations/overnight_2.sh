#!/usr/bin/env bash
# Overnight run 2, with the latent class check: Simulation 2 at r = 0.5, Simulation 3, Simulation 1.
# Run in tmux:  bash ~/pairD_eval/scripts/overnight_2.sh
# A grid that fails does not stop the next one; each start and end is written to overnight_2.log.
set -u
E=~/pairD_eval
LOG=$E/overnight_2.log
echo "$(date '+%F %T') overnight_2 started" >> "$LOG"

(
  mkdir -p "$E/test1/lca_r05/C1" && cd "$E/test1/lca_r05/C1" || exit 1
  echo "$(date '+%F %T') Simulation 2 r = 0.5 started" >> "$LOG"
  python3.14 "$E/scripts/run_test1_grid.py" -t "$E/test1/inputs/tables/test1_table_C1.csv" \
      --hpo-terms "$E/inputs/hpo_terms.json" --paird-args="--lca" \
      --disease D1S0 D2S2 D3S3 --n 90 180 --k 6 --reporting 0.5 --layout dispersed --substitution 0.25 \
      --distance dphenome -o lca_r05_results.csv --raw lca_r05_iterations.csv 2>&1 | tee lca_r05.log
  echo "$(date '+%F %T') Simulation 2 r = 0.5 ended (exit ${PIPESTATUS[0]})" >> "$LOG"
)

(
  P=$E/test2/min_dist_0/pairs_triple
  mkdir -p "$E/test2/lca" && cd "$E/test2/lca" || exit 1
  echo "$(date '+%F %T') Simulation 3 started" >> "$LOG"
  python3.14 "$E/scripts/run_test2_grid.py" \
      --diseases "$P/symptom_list_marfan.txt" "$P/symptom_list_wilson.txt" "$P/symptom_list_alagille.txt" \
      --hpo-terms "$P/hpo_terms.json" --distance dphenome --paird-args="--lca" 2>&1 | tee test2_lca.log
  echo "$(date '+%F %T') Simulation 3 ended (exit ${PIPESTATUS[0]})" >> "$LOG"
)

(
  mkdir -p "$E/test3/lca" && cd "$E/test3/lca" || exit 1
  echo "$(date '+%F %T') Simulation 1 started" >> "$LOG"
  python3.14 "$E/test3/run_test3_grid.py" --diseases "$E"/test3/phenomizer/symptom_list_*.csv \
      --hpo-terms "$E/test3/hpo_terms.json" --hpoa "$E/test3/phenotype.hpoa" \
      --distance dphenome --paird-args="--lca" 2>&1 | tee test3_lca.log
  echo "$(date '+%F %T') Simulation 1 ended (exit ${PIPESTATUS[0]})" >> "$LOG"
)
echo "$(date '+%F %T') overnight_2 finished" >> "$LOG"
