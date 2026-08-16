#!/usr/bin/env bash
# Are the DPO labels reproducible? 65.6% of round 2's pairs were decided by a 2-4 playout gap
# out of 16, which is ~1 SE of playout noise. Selection is deterministic given the traces, so
# re-running with a different --seed re-measures THE SAME branch points with fresh playouts.
#   arm A  same 16 playouts, new seed  -> how often does the label flip?
#   arm B  64 playouts on the top 5000 -> same CPU as 20000x16; yield + verdict change
set -u
say() { echo "[lab $(date -u +%m-%d_%H:%M:%S)] $*"; }
cd /root/ptcg/repo
T=/root/traces_r2.s0.jsonl.gz,/root/traces_r2.s1.jsonl.gz,/root/traces_r2.s2.jsonl.gz
COMMON="--traces $T --per-game 15 --margin-min 0.01 --workers 24"

say "arm A: replicate at 16 playouts, seed 22000"
CUDA_VISIBLE_DEVICES= PYTHONPATH=cg-lib nice -n 10 python3 tools/dpo_branch.py $COMMON \
  --budget 20000 --playouts 16 --seed 22000 --out /root/lab_rep16.jsonl.gz 2>&1 | tail -9

say "arm B: 64 playouts on the top 5000 (same total playout budget)"
CUDA_VISIBLE_DEVICES= PYTHONPATH=cg-lib nice -n 10 python3 tools/dpo_branch.py $COMMON \
  --budget 5000 --playouts 64 --seed 33000 --out /root/lab_rep64.jsonl.gz 2>&1 | tail -9

say "compare"
python3 - <<'PY'
import gzip, json, collections

def load(fn):
    d = {}
    for line in gzip.open(fn, "rt"):
        r = json.loads(line)
        d[(r["deck"], r["seed"], r["t"])] = r
    return d

base = load("/root/dpo_r2b.jsonl.gz")      # the labels round 2 actually trained on
for name, fn in (("A 16pl new seed", "/root/lab_rep16.jsonl.gz"),
                 ("B 64pl", "/root/lab_rep64.jsonl.gz")):
    rep = load(fn)
    both = set(base) & set(rep)
    agree = sum(1 for k in both if base[k]["tw"] == rep[k]["tw"])
    print("[%s] pairs %d | overlap with round 2's %d" % (name, len(rep), len(both)))
    if both:
        print("     LABEL AGREEMENT %.1f%%  (%d flipped)" % (100*agree/len(both), len(both)-agree))
    # a point that yielded a pair in one run and not the other is also instability
    only_b = len(set(base) - set(rep)); only_r = len(set(rep) - set(base))
    print("     pair in round2 only %d | in this run only %d" % (only_b, only_r))
    # agreement as a function of how decisive round 2's gap was
    by = collections.defaultdict(lambda: [0, 0])
    for k in both:
        u = round((base[k]["qw"] - base[k]["ql"]) * 8)
        by[min(u, 8)][0] += 1
        by[min(u, 8)][1] += (base[k]["tw"] == rep[k]["tw"])
    print("     agreement by round2's gap (playout units):")
    for u in sorted(by):
        n, a = by[u]
        print("        %s%d units  n=%5d  agree %.1f%%" % (">=" if u == 8 else "  ", u, n, 100*a/n))
PY
say LABTEST_DONE
