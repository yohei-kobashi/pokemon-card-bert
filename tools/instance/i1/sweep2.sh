#!/usr/bin/env bash
set -u
cd /root/ptcg/repo
echo "floor (target entropy) = 0.593"
for cfg in "3e-5 8 8" "3e-5 8 1" "1e-4 8 1" "3e-4 8 1" "1e-4 30 1"; do
  set -- $cfg
  printf "lr=%-6s epochs=%-3s accum=%-2s  " "$1" "$2" "$3"
  PYTHONPATH=cg-lib python3 tools/dusk_plan_train.py --data /root/rl/plan_r1.jsonl.gz \
    --model /root/out/d41_r8 --out /root/out/plan_probe --probe \
    --lr "$1" --epochs "$2" --accum "$3" 2>&1 | grep -aE "^\[probe\]|^FINAL" | tr '\n' ' '
  echo
done
echo SWEEP2_DONE
