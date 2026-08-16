#!/usr/bin/env bash
# v37 = v36 continued on WINNER-ONLY rows in v36's own prompt format.
#
# Why this and not a merge. v36 is a sequential continuation of v35, so exactly ONE direction
# exists in weight space (tau = v36 - v35) and every merge method collapses onto it: task
# arithmetic IS lerp with one task vector, TIES has no sign conflict to resolve with one
# vector, DARE degenerates to random sparsification of tau, and SLERP == lerp at 0.32%
# separation. The measured lerp scan was monotone in live-weighted score (65.1 -> 62.6 ->
# 61.3), so its argmax is alpha=0, i.e. v35 itself. Layer-wise/evolutionary merging is the one
# family with real extra degrees of freedom, but it needs a cheap fitness and ours costs ~12
# min per candidate with NO usable surrogate -- top1 has now disagreed with win rate twice.
#
# What this run does instead. The two effects of --sides winner were separated by measurement:
# it starves weak decks of data (5.01x spread; both-sides fixed it, mega_lucario +5.1pt) AND
# it filters for winning lines (removing it cost crustle_stall -4.0pt). v36 already holds the
# coverage fix in its weights. Applying the filter ON TOP of it is a new direction, not a
# point on the old segment -- the thing a merge structurally cannot give us.
#
# --cap-deck -1 is REQUIRED here. A matchup cap cannot balance winner-only data (it is an
# upper bound, and a losing matchup has too few winning rows to reach it): measured, this data
# at --cap-matchup 320 leaves per-deck 7,671..19,840 = 2.59x, worse than v35's 1.50x and far
# from v36's 1.004x. Without it a crustle_stall gain would be unattributable -- better data,
# or simply 2.6x more of it? With it the sample is 1.05x flat, so only the FILTER varies.
#
# Same hyperparameters as the v35->v36 step so the comparison stays interpretable.
set -u
cd /root/ptcg/repo
export PYTHONPATH=cg-lib
DATA=/root/ptcg/repo/data/rerank/curengine_0724_v36w.rerank.jsonl.gz,/root/ptcg/repo/data/rerank/v34_full_v36w.rerank.jsonl.gz
echo "=== v37 train start $(date -u) ==="
python3 tools/train_rerank.py --data "$DATA" \
  --model /root/out/rerank_gte_v36 --out /root/out/rerank_gte_v37 \
  --cap-matchup 320 --cap-deck -1 --max-samples 250000 \
  --max-len 640 --pair-batch 256 --accum 4 --grad-ckpt \
  --drop-deck 0 --drop-id 0.25 --emb-lr-mult 3 --lr 2e-5 --deadline-h 3
echo "=== v37 train done $(date -u) ==="

# crustle_stall against the LIVE field (the submission decision) and mega_lucario on the
# 3-opponent grid (did the coverage gain survive the filter?), both at 300 games/cell.
DECKS="crustle_stall" OPPS="alakazam_nz marnie_grimmsnarl archaludon alakazam cynthia_garchomp crustle dragapult" \
  tools/eval_rerank_par.sh /root/out/wr_v37_live /root/out/rerank_gte_v37 torch "" 8 300 1000000 none remaining 1
DECKS="mega_lucario" OPPS="alakazam crustle dragapult" \
  tools/eval_rerank_par.sh /root/out/wr_v37_ml /root/out/rerank_gte_v37 torch "" 8 300 1000000 none remaining 1
echo "=== V37_ALL_DONE $(date -u) ==="
