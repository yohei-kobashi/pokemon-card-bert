#!/usr/bin/env bash
# Pull the round-1 traces from instance2, branch them, ship the pairs back.
set -u
say() { echo "[br1 $(date -u +%m-%d_%H:%M:%S)] $*"; }
I2="-i /root/.ssh/id_i2 -o IdentitiesOnly=yes -o StrictHostKeyChecking=no"
scp -q $I2 -P 19839 "root@175.155.64.145:/root/traces_r2.s*.jsonl.gz" /root/ \
  || { say "trace pull FAILED"; exit 1; }
say "traces pulled: $(ls -la /root/traces_r2.s*.jsonl.gz | wc -l) files"
cd /root/ptcg/repo
CUDA_VISIBLE_DEVICES= PYTHONPATH=cg-lib nice -n 5 python3 tools/dpo_branch.py \
  --traces /root/traces_r2.s0.jsonl.gz,/root/traces_r2.s1.jsonl.gz,/root/traces_r2.s2.jsonl.gz \
  --budget 20000 --per-game 15 --margin-min 0.01 --playouts 16 --workers 36 \
  --out /root/dpo_r2.jsonl.gz || { say "branch FAILED"; exit 1; }
N=$(zcat /root/dpo_r2.jsonl.gz | wc -l)
say "shipping $N pairs"
scp -q $I2 -P 19839 /root/dpo_r2.jsonl.gz root@175.155.64.145:/root/dpo_r2.jsonl.gz \
  && say "instance2 has dpo_r2 ($N pairs)" || say "SHIP FAILED -- kept at /root/dpo_r2.jsonl.gz"
say BRANCH_R1_DONE
