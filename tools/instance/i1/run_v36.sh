#!/usr/bin/env bash
# v36: retrain the reranker on the REPAIRED data.
#
# Data (12,539,524 records available, balanced down to ~1.25M by --cap-matchup):
#   --label heuristic   explored steps carried the move engine_v2 REFUSED (below-chance rows)
#   --sides both        winner-only threw away half the engine's decisions
#   --deck-mode remaining + --deck-shuffle   DECK[] was a memorisable fixed fingerprint
#
# Training:
#   --drop-deck 0   segment dropout TEACHES invariance to DECK; that was a measurement tool
#                   and swapDECK now measures reliance without the training pressure
#   --drop-id 0.25  ID ME really is redundant given DECK[], so keep forcing the model off it
#   from v35, not v34: only v35's embeddings can distinguish two decklists (cos 0.71 vs 0.9999)
set -u
cd /root/ptcg/repo
export PYTHONPATH=cg-lib
DATA=/root/data/rerank/curengine_0724_v36.rerank.jsonl.gz,/root/data/rerank/v34_full_v36.rerank.jsonl.gz
ROWS=1250000

if [ ! -f /root/out/eval2k_v36.json ]; then
  echo "=== building eval split $(date -u) ==="
  python3 tools/ablate_rerank.py --data "$DATA" --cache /root/out/eval2k_v36.json \
    --cap-matchup 320 --max-samples "$ROWS" || exit 1
fi

echo "=== training $(date -u) ==="
python3 tools/train_rerank.py --data "$DATA" \
  --model /root/out/rerank_gte_v35 --out /root/out/rerank_gte_v36 \
  --eval-file /root/out/eval2k_v36.json \
  --cap-matchup 320 --max-samples "$ROWS" \
  --max-len 640 --pair-batch 256 --accum 4 --grad-ckpt \
  --drop-deck 0 --drop-id 0.25 --emb-lr-mult 3 --lr 2e-5 --deadline-h 14

echo "=== win rate v36 $(date -u) ==="
tools/eval_rerank_par.sh /root/out/wr_v36 /root/out/rerank_gte_v36 torch "" 8 60 1000000 none remaining 1
echo "=== V36_ALL_DONE ==="
