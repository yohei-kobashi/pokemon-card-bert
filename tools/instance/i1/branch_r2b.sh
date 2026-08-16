#!/usr/bin/env bash
# Re-branch round 2's ALREADY-PULLED traces with seat-fair budgeting, and ship the pairs.
# Rounds 1-2's pairs were 75% first-seat because the lowest-margin cut ignores the seat and
# the policy is less certain moving first; round 1's whole gate gain was seat0.
set -u
say() { echo "[br2b $(date -u +%m-%d_%H:%M:%S)] $*"; }
I2="-i /root/.ssh/id_i2 -o IdentitiesOnly=yes -o StrictHostKeyChecking=no"
for f in /root/traces_r2.s0.jsonl.gz /root/traces_r2.s1.jsonl.gz /root/traces_r2.s2.jsonl.gz; do
  [ -f "$f" ] || { say "missing $f"; exit 1; }
done
say "reusing 3 trace shards"
cd /root/ptcg/repo
CUDA_VISIBLE_DEVICES= PYTHONPATH=cg-lib nice -n 5 python3 tools/dpo_branch.py \
  --traces /root/traces_r2.s0.jsonl.gz,/root/traces_r2.s1.jsonl.gz,/root/traces_r2.s2.jsonl.gz \
  --budget 20000 --per-game 15 --margin-min 0.01 --playouts 16 --workers 36 \
  --out /root/dpo_r2b.jsonl.gz || { say "branch FAILED"; exit 1; }
python3 - <<'PY'
import gzip, json, collections
c = collections.Counter()
for line in gzip.open("/root/dpo_r2b.jsonl.gz", "rt"):
    c["s%d" % json.loads(line).get("seat", -1)] += 1
n = sum(c.values())
print("[br2b] pair seat split: " + " ".join("%s %d (%.1f%%)" % (k, v, 100*v/n) for k, v in sorted(c.items())))
PY
N=$(zcat /root/dpo_r2b.jsonl.gz | wc -l)
say "shipping $N pairs"
scp -q $I2 -P 19839 /root/dpo_r2b.jsonl.gz root@175.155.64.145:/root/dpo_r2b.jsonl.gz \
  && say "instance2 has dpo_r2b ($N pairs)" || { say "SHIP FAILED -- kept local"; exit 1; }
say BRANCH_R2B_DONE
