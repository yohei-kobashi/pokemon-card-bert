#!/bin/bash
# full vs two independent 50% subsamples of the SAME rollout, from the SAME start.
set -u
cd /root/ptcg/repo
export PYTHONPATH=cg-lib PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RO=/root/out/rlA/A_r1.jsonl.gz
M=/root/out/rerank_gte_v37
for spec in "full:1.0:1" "half_a:0.5:11" "half_b:0.5:22"; do
  N=${spec%%:*}; R=$(echo $spec | cut -d: -f2); SD=$(echo $spec | cut -d: -f3)
  [ -f /root/out/frac_$N/config.json ] && { echo "skip $N"; continue; }
  S=$(date +%s)
  python3 tools/rl_train.py --rollout $RO --model $M --out /root/out/frac_$N \
      --grad-ckpt --win-boost --decision-frac $R --decision-seed $SD > /root/frac_$N.log 2>&1
  echo "$N (frac=$R seed=$SD): WALL $(( $(date +%s) - S )) s"
done
echo "FRAC_EXP_DONE"
