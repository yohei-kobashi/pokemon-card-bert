#!/usr/bin/env bash
# The one-row test settles it: at lr 1e-5 the trainer drives a single row from 0.332 to 0.007,
# so the loss and the optimizer are fine. The probe diverged because every arm of the previous
# sweep sat in the same too-high band (3e-5 / 1e-4 / 3e-4) at an effective batch size of ONE.
# Search DOWNWARD, and give accumulation a real batch to work with.
set -u
cd /root/ptcg/repo
export PYTHONPATH=cg-lib
echo "floor 0.278 | 300 rows | 30 epochs = 9000 steps"
for cfg in "1e-5 1" "1e-5 8" "5e-6 8" "2e-5 8" "3e-5 16"; do
  set -- $cfg
  printf "lr=%-6s accum=%-3s  " "$1" "$2"
  python3 tools/dusk_plan_train.py --data /root/rl/plan_r4.jsonl.gz --model /root/out/d41_r8 \
    --out /root/out/plan_probe --probe --lr "$1" --epochs 30 --accum "$2" 2>&1 \
    | grep -aE "^FINAL|^PROBE" | tr "\n" " "
  echo
done
echo SWEEP3_DONE
