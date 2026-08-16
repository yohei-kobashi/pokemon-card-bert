#!/usr/bin/env bash
# Everything that must happen the moment v35 stops, in the user's stated order:
#   1. WIN RATE v34 vs v35 (the deliverable metric)
#   2. top1 on the SAME old eval split (apples-to-apples vs v34's 65.7%)
#   3. the swapDECK/turn ablation on the new split (does v35 read the deck?)
# Win rate goes first because it is the number the submit/retrain decision rests on.
set -u
cd /root/ptcg/repo
export PYTHONPATH=cg-lib

while pgrep -f "tools/train_rerank" > /dev/null; do sleep 30; done
echo "=== training stopped $(date -u) ==="
sleep 20                                   # let CUDA memory actually free

for M in v34 v35; do
  echo "=== WIN RATE rerank_gte_$M ($(date -u)) ==="
  tools/eval_rerank_par.sh "/root/out/wr_$M" "/root/out/rerank_gte_$M" torch "" 8 60 1000000 none static
done

echo "=== TOP1 on the OLD (v34) eval split -- comparable to 65.7% ==="
python3 tools/ablate_rerank.py --cache /root/out/eval2k_v34.json --by-turn --pair-batch 128 \
  --models /root/out/rerank_gte_v34,/root/out/rerank_gte_v35

echo "=== ABLATION on the NEW balanced split ==="
python3 tools/ablate_rerank.py --cache /root/out/eval2k_v2.json --by-turn --pair-batch 128 \
  --models /root/out/rerank_gte_v34,/root/out/rerank_gte_v35

echo "=== AFTER_V35_ALL_DONE ==="
