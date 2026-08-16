#!/usr/bin/env bash
# Rebuild the plan data with the three audited rules, then probe at 30 epochs.
# The audit changed what "correct" means, so the old file would train the old mistakes:
#   energy_line  -- {D} belongs on Munkidori, not on a Dreepy
#   evolve_line  -- keep Drakloak drawing unless the Dragapult can actually attack
#   boss_damaged -- cash BANKED damage; a fresh 70 HP Dreepy is in range and worth nothing
set -u
cd /root/ptcg/repo
export PYTHONPATH=cg-lib
python3 tools/dusk_plan_data.py --traces "$(ls /root/traces_r4.s*.jsonl.gz | paste -sd,)" \
  --mirror-so /root/ptcg/repo/data/kaggle_engine_ext/libcg_mirror.so \
  --out /root/rl/plan_r2.jsonl.gz 2>&1 | head -4
python3 /root/floor.py
for cfg in "1e-4 30" "5e-5 30" "1e-4 8"; do
  set -- $cfg
  printf "lr=%-6s epochs=%-3s  " "$1" "$2"
  python3 tools/dusk_plan_train.py --data /root/rl/plan_r2.jsonl.gz --model /root/out/d41_r8 \
    --out /root/out/plan_probe --probe --lr "$1" --epochs "$2" --accum 1 2>&1 \
    | grep -aE "^\[probe\]|^FINAL|^PROBE" | tr "\n" " "
  echo
done
echo REBUILD_PROBE_DONE
