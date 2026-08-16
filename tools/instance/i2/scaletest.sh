#!/bin/bash
# Does the step time scale with sequence length at all?
#
# Length grouping cut padded tokens 22.5% and bought 1.9% of wall clock, so the run is not paying
# for tokens the way a compute-bound run would. Forcing max_length down truncates the prompt --
# which destroys the labels, and would be a serious bug in a real run -- but this measures TIME
# ONLY, and it separates the two candidate worlds cleanly: time roughly proportional to length
# means compute-bound and the A/B was confounded; time flat means a fixed per-step cost dominates
# and that cost is where any real speed-up has to come from.
set -u
cd /root/ptcg/repo
for L in 128 256 512 896; do
  echo "=== maxlen $L ==="
  timeout 1800 python3 tools/instance/sft_teacher.py --domain-tokens \
    --action-vocab data/action_vocab_v39.json \
    --model unsloth/Qwen3-4B-Base --data data/sft/v39_dag005.jsonl.gz \
    --out /root/out/scale_$L --limit 40000 --eval-n 0 --steps 30 --maxlen $L \
    --save-steps 100000 --bsz 8 --accum 4 --group-by-length 2>&1 \
    | grep -E "^\[done\]|^\[peak\]|out of memory"
  rm -rf /root/out/scale_$L
done
echo "SCALETEST DONE"
