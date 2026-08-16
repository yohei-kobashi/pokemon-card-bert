#!/usr/bin/env bash
# Round 3 pairs: SAME traces, SAME 20,000 seat-fair branch points as round 2 -- only the
# evidence behind each label changes (16 -> 64 playouts). That makes round 3 a controlled test
# of the label-noise hypothesis rather than another mixed change.
set -u
say() { echo "[br3 $(date -u +%m-%d_%H:%M:%S)] $*"; }
I2="-i /root/.ssh/id_i2 -o IdentitiesOnly=yes -o StrictHostKeyChecking=no"
cd /root/ptcg/repo
CUDA_VISIBLE_DEVICES= PYTHONPATH=cg-lib nice -n 10 python3 tools/dpo_branch.py \
  --traces /root/traces_r2.s0.jsonl.gz,/root/traces_r2.s1.jsonl.gz,/root/traces_r2.s2.jsonl.gz \
  --budget 20000 --per-game 15 --margin-min 0.01 --playouts 64 --workers 30 \
  --out /root/dpo_r3.jsonl.gz || { say "branch FAILED"; exit 1; }
python3 - <<'PY'
import gzip, json, collections, statistics as st
rs = [json.loads(l) for l in gzip.open("/root/dpo_r3.jsonl.gz", "rt")]
c = collections.Counter("s%d" % r["seat"] for r in rs)
dq = sorted(r["qw"] - r["ql"] for r in rs)
print("[br3] %d pairs | seat %s | dQ mean %.3f p50 %.3f p90 %.3f | pl=%s"
      % (len(rs), dict(c), st.mean(dq), dq[len(dq)//2], dq[int(.9*len(dq))],
         sorted({r.get("pl") for r in rs})))
PY
N=$(zcat /root/dpo_r3.jsonl.gz | wc -l)
say "shipping $N pairs"
scp -q $I2 -P 19839 /root/dpo_r3.jsonl.gz root@175.155.64.145:/root/dpo_r3.jsonl.gz \
  && say "instance2 has dpo_r3 ($N pairs)" || { say "SHIP FAILED"; exit 1; }
say BRANCH_R3_DONE
