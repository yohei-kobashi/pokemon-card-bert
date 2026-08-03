#!/bin/bash
# Time the SFT config levers on the REAL training path, not a mock.
#
# Each config runs tools/instance/sft_teacher.py itself for a few dozen steps, so whatever is
# measured here is exactly what a full run does -- the alternative (a standalone timing harness)
# measures a second implementation that can drift from the one that trains.
#
# EFFECTIVE BATCH IS HELD AT 32 across the batch-size rows (8x4, 16x2, 32x1). Comparing rec/s
# across different effective batches would conflate throughput with a change in the optimisation
# itself, and only throughput is under test here.
#
# READ flops/s, NOT samples/s. LengthGroupedSampler deliberately puts the longest megabatch
# first, so a 40-step run sees the longest ~3% of the data and its samples/s is not comparable
# with an unsorted run's -- measured, base_b8 8.014 vs group_b8 7.523 samples/s while both did
# 6.4e13 flops/s, i.e. identical utilisation on different-sized work. Utilisation is what a
# 40-step run can measure; the padding saving is a counting fact (1.30x -> 1.00x padded tokens
# at batch 8, from tools/instance/measure_lengths.py) that only shows up over a full epoch.
#
# maxlen is NOT a row. Padding is dynamic (each batch pads to its own longest member), so
# max_length only truncates and costs nothing when nothing exceeds it -- measured max is 814
# tokens, so 896 is set everywhere as headroom rather than swept.
set -u
REPO=/root/ptcg/repo
MODEL=${MODEL:-unsloth/Qwen3-4B-Base}
DATA=${DATA:-/root/ptcg/repo/data/sft/v39_dag005.jsonl.gz}
VOCAB=${VOCAB:-/root/ptcg/repo/data/action_vocab_v39.json}
STEPS=${STEPS:-40}
LOG=/root/bench_sft.log
cd "$REPO"
exec >> "$LOG" 2>&1
say() { echo "[bench $(date -u +%H:%M:%S)] $*"; }

run() {  # name, extra args
  local name="$1"; shift
  say "=== $name : $* ==="
  local out=/root/out/bench_$name
  rm -rf "$out"
  timeout 3600 python3 tools/instance/sft_teacher.py --domain-tokens \
      --action-vocab "$VOCAB" \
      --model "$MODEL" --data "$DATA" --out "$out" \
      --limit 40000 --eval-n 0 --steps "$STEPS" --maxlen 896 --save-steps 100000 \
      "$@" 2>&1 | grep -E "^\[done\]|^\[peak\]|^\[peft\]|^\[len\]|^\[action\]|OutOfMemory|out of memory" \
      | head -8
  rm -rf "$out"
}

say "############ bench start | model $MODEL | steps $STEPS ############"
run base_b8            --bsz 8  --accum 4
run group_b8           --bsz 8  --accum 4 --group-by-length
run group_b16          --bsz 16 --accum 2 --group-by-length
run group_b32          --bsz 32 --accum 1 --group-by-length
run group_b32_nockpt   --bsz 32 --accum 1 --group-by-length --no-grad-ckpt
say "############ bench done ############"
