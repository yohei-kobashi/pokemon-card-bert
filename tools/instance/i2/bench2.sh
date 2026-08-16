#!/bin/bash
# What is still on the table for instance2's SFT, measured on the real training path.
#
# Levers ALREADY SWEPT and found inert (tools/instance/bench_sft.sh, on instance1's 24 GB 4090):
# batch size at fixed effective batch, gradient checkpointing, max_length (padding is dynamic and
# the measured longest sample is 814 tokens), length grouping (1.9%, not the 30% I claimed).
#
# What that sweep COULD NOT see: instance2 has 47.4 GiB against the 4090's 24, and the scheme-A
# run peaked at 30.8 GiB with checkpointing ON and batch 8. Configurations that simply did not
# fit on the 4090 fit here, so "inert" was measured over a range that stopped early.
#
# READ flops/s, NOT samples/s: LengthGroupedSampler puts the longest megabatch first, so a short
# run sees the longest few percent of the data and its samples/s is not comparable across runs
# with different batch sizes.
#
# CONFOUND, held constant: attach_q3 is using 90 of the 112 cores throughout, so every row pays
# the same dataloader contention. These numbers are for comparing rows, not for projecting a
# full run's wall clock.
set -u
REPO=/root/ptcg/repo
MODEL=unsloth/Qwen3-4B-Base
DATA=/root/ptcg/repo/data/sft/cf_b_r2.jsonl.gz
VOCAB=/root/ptcg/repo/data/cardfirst_b_v39.json
STEPS=${STEPS:-25}
LOG=/root/bench2.log
cd "$REPO"
exec >> "$LOG" 2>&1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
say() { echo "[bench2 $(date -u +%H:%M:%S)] $*"; }

run() {
  local name="$1"; shift
  say "=== $name : $* ==="
  rm -rf "/root/out/b2_$name"
  timeout 2400 python3 tools/instance/sft_teacher.py \
      --model "$MODEL" --data "$DATA" --domain-tokens --card-first "$VOCAB" \
      --init-from /root/out/qwen3_4b_cf1 \
      --out "/root/out/b2_$name" --limit 4000 --eval-n 0 --steps "$STEPS" \
      --maxlen 896 --group-by-length --save-steps 1000000 "$@" 2>&1 \
    | grep -E "train_runtime|train_samples_per_second|total_flos|\[peak\]|OutOfMemory|Error" \
    | tail -4
  rm -rf "/root/out/b2_$name"
}

run base_b8      --bsz 8  --accum 4
run b16          --bsz 16 --accum 2
run b16_nockpt   --bsz 16 --accum 2 --no-grad-ckpt
run b32          --bsz 32 --accum 1
say "BENCH2 DONE"
