#!/usr/bin/env bash
# Round 4 branch: the first cross-deck traces, and the first with the seat-perspective fix.
# 16 workers, not 30 -- the dusknoir generator owns 28 of instance1's 61 effective cores.
set -u
say() { echo "[br5 $(date -u +%m-%d_%H:%M:%S)] $*"; }
I2="-i /root/.ssh/id_i2 -o IdentitiesOnly=yes -o StrictHostKeyChecking=no"
cd /root/ptcg/repo
# Traces are relayed through the operator machine: vast periodically rewrites instance2's
# authorized_keys and drops instance1's key, which is what killed the first attempt here.
ls /root/traces_r5.s*.jsonl.gz >/dev/null 2>&1 || { say "no traces staged"; exit 1; }
say "pulled $(ls /root/traces_r5.s*.jsonl.gz | wc -l) trace shards"
CUDA_VISIBLE_DEVICES= PYTHONPATH=cg-lib nice -n 10 python3 tools/dpo_branch.py \
  --traces "$(ls /root/traces_r5.s*.jsonl.gz | paste -sd,)" \
  --budget 20000 --per-game 15 --margin-min 0.01 --playouts 32 --workers 16 \
  --out /root/dpo_r5.jsonl.gz || { say "branch FAILED"; exit 1; }
python3 -c "
import gzip, json, collections, statistics as st
rs=[json.loads(l) for l in gzip.open('/root/dpo_r5.jsonl.gz','rt')]
c=collections.Counter(r['seat'] for r in rs); o=collections.Counter(r['opp'] for r in rs)
p=collections.Counter(r['deck'] for r in rs)
dq=sorted(r['qw']-r['ql'] for r in rs)
print('[br5] %d pairs | seat %s | dQ p50 %.3f' % (len(rs), dict(c), dq[len(dq)//2]))
print('[br5] piloting: %s' % dict(p.most_common(4)))
print('[br5] top opponents: %s' % dict(o.most_common(5)))
"
N=$(zcat /root/dpo_r5.jsonl.gz | wc -l)
scp -q $I2 -P 19839 /root/dpo_r5.jsonl.gz root@175.155.64.145:/root/dpo_r5.jsonl.gz \
  && say "instance2 has dpo_r5 ($N pairs)" || { say "SHIP FAILED"; exit 1; }
say BRANCH_R5_DONE
