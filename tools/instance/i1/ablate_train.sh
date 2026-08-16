#!/usr/bin/env bash
# Why did v36 lose ground where engine_v2 is weak? Three arms, one factor each.
#
# v36 changed FIVE things at once vs v35 (label fix, --sides both, --deck-shuffle,
# --deck-mode remaining, 3.7x records), so its regression cannot be attributed. These arms all
# start from the SAME v35 checkpoint, train on the SAME number of records with the SAME
# hyperparameters, and differ in exactly one property:
#
#   A  both sides + shuffled   (= the v36 recipe)          -- the control
#   B  WINNER only + shuffled  (isolates --sides both)
#   C  both sides + FIXED order(isolates --deck-shuffle)
#
# Read-out is NOT win rate. The mechanism is already localised: deferring `attach` decisions
# to engine_v2 recovers +11.4pt of v36's alakazam_nz deficit while deferring `retreat`
# recovers nothing, and v36's attach deficit vs the engine is -6.7pp where v35's is -2.6pp.
# The attach deficit is measured over ~14k decisions per arm (SE ~0.3pp) instead of 300 games
# (SE 2.7pt), so it resolves the arms an order of magnitude better than win rate can.
#
# No --eval-file: arm C renders DECK[] differently, so a shared eval split would neither
# dedup against training nor give comparable top1. Each arm takes its own internal split, and
# top1 is not the read-out anyway.
set -u
cd /root/ptcg/repo
export PYTHONPATH=cg-lib
ROWS=300000
COMMON="--model /root/out/rerank_gte_v35 --cap-matchup 320 --max-samples $ROWS \
  --max-len 640 --pair-batch 256 --accum 4 --grad-ckpt \
  --drop-deck 0 --drop-id 0.25 --emb-lr-mult 3 --lr 2e-5 --deadline-h 4"

A_DATA=/root/data/rerank/curengine_0724_v36.rerank.jsonl.gz,/root/data/rerank/v34_full_v36.rerank.jsonl.gz
B_DATA=/root/ptcg/repo/data/rerank/curengine_0724_v36w.rerank.jsonl.gz,/root/ptcg/repo/data/rerank/v34_full_v36w.rerank.jsonl.gz
C_DATA=/root/data/rerank/curengine_0724_v36_noshuf.rerank.jsonl.gz,/root/data/rerank/v34_full_v36_noshuf.rerank.jsonl.gz

run () {                       # $1 arm name, $2 data
  local arm=$1 data=$2
  if [ -f "/root/out/abl_$arm/config.json" ]; then
    echo "=== arm $arm already trained, skipping ==="
    return
  fi
  echo "=== arm $arm start $(date -u) ==="
  python3 tools/train_rerank.py --data "$data" --out "/root/out/abl_$arm" $COMMON \
    > "/root/abl_$arm.log" 2>&1
  echo "=== arm $arm done $(date -u) ==="
  tail -2 "/root/abl_$arm.log"
}

run A "$A_DATA"
run B "$B_DATA"
run C "$C_DATA"
echo "=== ABLATE_TRAIN_ALL_DONE $(date -u) ==="
