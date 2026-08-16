#!/usr/bin/env bash
# Which of the two remaining suspects blocks memorisation: the LR, or the margin term?
# Same 1,800 rows as the bf16/fp32 probe (both of those were FLAT, so precision is excluded).
set -u
cd /root/ptcg/repo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=data/rerank/deberta41_r4.jsonl.gz
[ -s "$D" ] || D=data/rerank/v41_base.jsonl.gz
for ARM in "1e-5 0.0 margin0" "5e-5 0.5 lr5e5" "1e-4 0.5 lr1e4" "3e-4 0.5 lr3e4"; do
  set -- $ARM
  rm -rf /root/out/probe_lr && cp -r /root/out/d41_r4 /root/out/probe_lr
  rm -f /root/out/probe_lr/rr_progress.json
  echo "===== ARM $3 (lr $1 margin $2) data $D ====="
  python3 tools/train_rerank.py --data $D --out /root/out/probe_lr --resume --fp32 \
    --rows 2000 --eval-n 200 --max-samples 36000 --deadline-h 0.25 \
    --lr $1 --margin-weight $2 --pair-batch 32 --accum 12 --max-len 512 --grad-ckpt 2>&1 \
    | grep -aE "^  step |^FINAL|out of memory|Error"
done
rm -rf /root/out/probe_lr
echo PROBE_SET_DONE
