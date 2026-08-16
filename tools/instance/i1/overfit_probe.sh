#!/usr/bin/env bash
# Can the trainer overfit 1,800 rows? That single question separates "the optimiser cannot move"
# from "the data has no more signal", and it is the test to run before any other plateau fix.
# Two arms, identical except for where the WEIGHTS live.
set -u
cd /root/ptcg/repo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=data/rerank/deberta41_r4.jsonl.gz
for ARM in bf16 fp32; do
  F=""; [ "$ARM" = fp32 ] && F="--fp32"
  rm -rf /root/out/probe_$ARM && cp -r /root/out/d41_r3 /root/out/probe_$ARM
  rm -f /root/out/probe_$ARM/rr_progress.json
  echo "===== ARM $ARM ====="
  python3 tools/train_rerank.py --data $D --out /root/out/probe_$ARM --resume $F \
    --rows 2000 --eval-n 200 --max-samples 54000 --deadline-h 0.5 \
    --lr 1e-5 --pair-batch 32 --accum 12 --max-len 512 --grad-ckpt 2>&1 \
    | grep -aE "^  step |^  \[eval\]|^FINAL|out of memory"
  rm -rf /root/out/probe_$ARM
done
