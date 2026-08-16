#!/bin/bash
# 38.6% of the per-sample time does not scale with sequence length. The prime suspect is
# unsloth's gradient checkpointing, which OFFLOADS activations to system RAM -- a PCIe round trip
# per layer per micro-batch, largely independent of how long the sequence is.
#
# An earlier run measured --no-grad-ckpt at +0.9%, which is implausibly small for removing a
# recompute, so the flag is suspected of not taking effect. PEAK VRAM SETTLES IT: without
# checkpointing the activations must stay resident and the number has to jump. If it does not
# move, the flag was ignored and that earlier +0.9% measured nothing.
set -u
cd /root/ptcg/repo
run() {
  echo "=== $1 ==="; shift
  timeout 1800 python3 tools/instance/sft_teacher.py --domain-tokens \
    --action-vocab data/action_vocab_v39.json \
    --model unsloth/Qwen3-4B-Base --data data/sft/v39_dag005.jsonl.gz \
    --out /root/out/ck --limit 40000 --eval-n 0 --steps 30 --maxlen 896 \
    --save-steps 100000 --bsz 8 --accum 4 --group-by-length "$@" 2>&1 \
    | grep -E "^\[done\]|^\[peak\]|out of memory"
  rm -rf /root/out/ck
}
run "checkpointing ON (unsloth offload)"
run "checkpointing OFF" --no-grad-ckpt
echo "CKPTTEST DONE"
