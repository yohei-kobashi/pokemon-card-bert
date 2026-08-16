#!/bin/bash
# v39 reranker training, with the two things the last two attempts lacked:
#   1. --grad-ckpt.  The v36/v37 recipe always used it; dropping it is why both attempts OOMed.
#      Lowering --pair-batch alone only delays the death until a batch of long rows comes up
#      (step 100 at pb=48, step 1700 at pb=24) -- length bucketing makes that arrival random.
#   2. A retry loop.  An OOM 6 hours in otherwise burns the night silently; on failure this
#      halves the batch and resumes from the last checkpoint.
cd /root/ptcg/repo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=/root/out/rerank_gte_v39
PB=32          # 32x768 = 24.6k token-slots vs the proven 48x640 = 30.7k, so below a known-good peak
ACC=12         # effective batch ~384, same as the v37 recipe
for TRY in 1 2 3; do
  RES=""
  [ -d "$OUT" ] && RES="--resume"
  echo "=== attempt $TRY: pair-batch $PB accum $ACC $RES  $(date -u +%H:%M:%S) ===" >> /root/train_v39.log
  python3 tools/train_rerank.py \
    --data data/rerank/v39_0731.rerank.jsonl.gz \
    --out "$OUT" --deadline-h 6 --max-samples 800000 \
    --pair-batch $PB --accum $ACC --lr 2e-5 --max-len 768 --eval-n 2000 \
    --grad-ckpt $RES >> /root/train_v39.log 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then echo "=== finished ok ===" >> /root/train_v39.log; exit 0; fi
  if ! tail -40 /root/train_v39.log | grep -qi "outofmemory"; then
    echo "=== died rc=$rc, NOT an OOM -- not retrying ===" >> /root/train_v39.log; exit $rc
  fi
  PB=$((PB / 2)); ACC=$((ACC * 2))
done
echo "=== gave up after 3 attempts ===" >> /root/train_v39.log
