#!/usr/bin/env bash
# Phase 1 of the merge test: which alpha, and which prompt format, on the ONE matchup that
# separates the endpoints.
#
# crustle_stall vs alakazam_nz is where v36 collapsed (22.3% vs v35's 34.0%, engine 47.7%) and
# it is the single most common live opponent (21.2% of the top-500 ladder), so it is both the
# most diagnostic cell and the one that matters most. A merge that does not move this cell
# cannot help the live-weighted score.
#
# FORMAT IS A FREE PARAMETER for a merged model: v35 was trained on static/no-shuffle DECK[],
# v36 on remaining/shuffled, and the merge belongs to neither. It is measured, not assumed --
# though the ablations say the model barely reads DECK[] at all (swapDECK -0.5pt), so the
# expectation is that format matters little and alpha does the work.
set -u
cd /root/ptcg/repo
export PYTHONPATH=cg-lib
G=${1:-300}
for a in 25 50 75; do
  for fmt in s r; do
    if [ "$fmt" = s ]; then MODE=static; SH=0; else MODE=remaining; SH=1; fi
    OUT=/root/out/mscan_a${a}_${fmt}
    [ -f "$OUT/crustle_stall__alakazam_nz.json" ] && { echo "skip a$a $fmt"; continue; }
    DECKS="crustle_stall" OPPS="alakazam_nz" \
      nohup tools/eval_rerank_par.sh "$OUT" /root/out/merge_a$a torch "" 8 "$G" 1000000 none $MODE $SH \
      > /root/out/mscan_a${a}_${fmt}.log 2>&1 &
  done
done
wait
echo "=== MERGE_SCAN_DONE $(date -u) ==="
