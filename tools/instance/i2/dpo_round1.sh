#!/usr/bin/env bash
# 4B DPO round 1: probe -> train -> 11x229 gate, paired against gate_r1 (i2_r7, same seeds).
set -u
REPO=/root/ptcg/repo
VOCAB=$REPO/data/cardfirst_b_v39.json
PAIRS=/root/dpo_r1.jsonl.gz
OUT=/root/out/dpo_r1
STATE=/root/loop_dpo
MIRROR_SO=$REPO/data/kaggle_engine_ext/libcg_mirror.so
mkdir -p "$STATE"
cd "$REPO"
say() { echo "[dpo1 $(date -u +%m-%d_%H:%M:%S)] $*"; }

# ---- overfit probe first. A trainer that cannot overfit 1k pairs must not spend a round;
# instance1 just spent 11 rounds learning that lesson the slow way.
say "probe start"
python3 tools/instance/dpo_teacher.py --data "$PAIRS" --init-from /root/out/i2_r7 \
  --card-first "$VOCAB" --out /root/out/dpo_probe --probe --lr 5e-5 > "$STATE/probe.log" 2>&1
grep -aE "PROBE" "$STATE/probe.log" || { say "probe crashed -- see $STATE/probe.log"; exit 1; }
grep -aq "PROBE OK" "$STATE/probe.log" || { say "PROBE FAILED -- stopping before the round"; exit 1; }

# ---- the real round: 3 epochs over ~4.5k pairs = ~420 optimizer updates at bsz 8 x accum 4.
# One epoch would be ~140 updates, which is the update starvation instance1 just diagnosed.
say "train start"
python3 tools/instance/dpo_teacher.py --data "$PAIRS" --init-from /root/out/i2_r7 \
  --card-first "$VOCAB" --out "$OUT" --epochs 3 --beta 0.1 --lr 5e-5 \
  > "$STATE/train.log" 2>&1 || { say "train FAILED"; tail -5 "$STATE/train.log"; exit 1; }
grep -aE "FINAL|saved" "$STATE/train.log" | tail -3
[ -f "$OUT/domain_embeddings.pt" ] || { say "STOP: no domain_embeddings.pt in $OUT"; exit 1; }

# ---- gate: same shards, seeds and .so as the stage1 gates, so gate_r1.json pairs exactly
say "gate start"
DECKS=$(python3 -c "import sys;sys.path.insert(0,\"tools\");import rl_config;print(\",\".join(rl_config.STAGE_C_TARGETS))")
python3 - "$DECKS" > "$STATE/shards.txt" <<PYX
import sys
d = sys.argv[1].split(",")
for i in range(3):
    print(" ".join("--deck " + x for x in d[i::3]))
PYX
j=0
while read -r DK; do
  [ -n "$DK" ] || continue
  PYTHONPATH=cg-lib nohup python3 tools/mirror_match.py $DK --a engine --b "qwen:$OUT" \
      --max-games 229 --mirror --seed 1 --mirror-so "$MIRROR_SO" \
      --out "$STATE/gate_dpo1.$j.json" > "$STATE/gate_dpo1.$j.log" 2>&1 &
  j=$((j+1))
done < "$STATE/shards.txt"
say "launched $j gate shards"
wait
python3 - <<PY
import json, math, statistics as st
d = {}
for k in range(3):
    try:
        d.update(json.load(open("/root/loop_dpo/gate_dpo1.%d.json" % k))["decks"])
    except Exception as e:
        print("shard %d unreadable: %s" % (k, e))
p = [v["p"] for v in d.values()]
print("[gate] dpo_r1 | %d decks | mean %.1f%% | median %.1f%% | below50 %d"
      % (len(d), 100*st.mean(p), 100*st.median(p), sum(1 for x in p if x < .5)))
q = json.load(open("/root/loop_stage1/gate_r1.json"))["decks"]
both = sorted(set(d) & set(q))
dd = [d[k]["p"] - q[k]["p"] for k in both]
m = st.mean(dd); se = st.stdev(dd)/math.sqrt(len(dd))
print("[paired vs i2_r7, %d decks] %+.4f +- %.4f  t %+.2f" % (len(both), m, se, m/se if se else 0))
for k in sorted(both, key=lambda k: d[k]["p"]-q[k]["p"]):
    print("   %-22s %5.1f%% -> %5.1f%%  (%+.1f)" % (k, 100*q[k]["p"], 100*d[k]["p"], 100*(d[k]["p"]-q[k]["p"])))
PY
say DPO_ROUND1_DONE
