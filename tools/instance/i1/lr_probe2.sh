#!/usr/bin/env bash
# lr_probe.sh, relaunched with a GPU gate. The first attempt collided with the round-5 screen:
# 8 mirror shards each hold a DeBERTa on the GPU (~23 GiB together), so every arm OOMed at
# startup. Wait for >= 8 GiB free -- the training phase holds ~5.5 GiB and leaves 18 -- and
# hold BETWEEN arms too, since a screen can start while an arm runs.
set -u
cd /root/ptcg/repo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=data/rerank/v41_base.jsonl.gz
wait_gpu() {
  local n=0
  while :; do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "$FREE" -ge 8000 ] && return 0
    n=$((n+1)); [ $((n % 10)) -eq 1 ] && echo "[gate] ${FREE} MiB free -- waiting"
    sleep 60
  done
}
for ARM in "1e-5 0.0 margin0" "5e-5 0.5 lr5e5" "1e-4 0.5 lr1e4" "3e-4 0.5 lr3e4"; do
  set -- $ARM
  wait_gpu
  rm -rf /root/out/probe_lr && cp -r /root/out/d41_r4 /root/out/probe_lr
  rm -f /root/out/probe_lr/rr_progress.json
  echo "===== ARM $3 (lr $1 margin $2) ====="
  python3 tools/train_rerank.py --data $D --out /root/out/probe_lr --resume --fp32 \
    --rows 2000 --eval-n 200 --max-samples 36000 --deadline-h 0.25 \
    --lr $1 --margin-weight $2 --pair-batch 32 --accum 12 --max-len 512 --grad-ckpt 2>&1 \
    | grep -aE "^  step |^FINAL|out of memory|Error"
done
rm -rf /root/out/probe_lr
echo PROBE_SET_DONE
