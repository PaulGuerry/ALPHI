#!/usr/bin/env bash
# Progress of the overnight runs:  bash ~/pairD_eval/scripts/progress.sh
# (or every 60 s:  watch -n 60 bash ~/pairD_eval/scripts/progress.sh)
E=~/pairD_eval
count() { [ -f "$1" ] && echo $(( $(wc -l < "$1") - 1 )) || echo "-"; }
echo "=== Grids started and ended"
cat "$E"/overnight_1.log "$E"/overnight_2.log 2>/dev/null | sort
echo
echo "=== ALPHI runs done (expected totals are approximate: extra cohorts near thresholds)"
printf "  %-28s %6s  %s\n" "Simulation 2 C1" "$(count $E/test1/lca/C1/lca_iterations.csv)" "of ~1,100-2,200"
printf "  %-28s %6s  %s\n" "Simulation 2 C2" "$(count $E/test1/lca/C2/lca_iterations.csv)" "of ~1,100-2,200"
printf "  %-28s %6s  %s\n" "Simulation 2 C3" "$(count $E/test1/lca/C3/lca_iterations.csv)" "of ~1,100-2,200"
printf "  %-28s %6s  %s\n" "Simulation 2, r = 0.5" "$(count $E/test1/lca_r05/C1/lca_r05_iterations.csv)" "of 30-60"
printf "  %-28s %6s  %s\n" "Simulation 3" "$(count $E/test2/lca/test2_iterations.csv)" "(searches: number varies)"
printf "  %-28s %6s  %s\n" "Simulation 1" "$(count $E/test3/lca/test3_iterations.csv)" "of 20 (880 patients each)"
echo
echo "=== Latest decisions of the latent class check"
grep -h "Decision:" $(ls -t "$E"/test1/lca/C*/lca.log "$E"/test1/lca_r05/C1/lca_r05.log "$E"/test2/lca/test2_lca.log \
    "$E"/test3/lca/test3_lca.log 2>/dev/null | head -1) 2>/dev/null | tail -3
echo
echo "=== Memory"
free -h | head -2
kills=$(dmesg 2>/dev/null | grep -c "Killed process")
[ -n "$kills" ] && echo "  processes killed for lack of memory since start-up: $kills (before tonight: 5)"
