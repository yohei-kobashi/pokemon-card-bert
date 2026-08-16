#!/usr/bin/env bash
# 4B DPO round 2: iterated DPO -- the reference advances with the policy (init-from dpo_r1),
# so beta anchors to round 1's policy, not to i2_r7. Gate is read BOTH ways:
# per-round vs gate_dpo1 (does this round help?) and cumulative vs gate_r1 (is RL ahead of i2_r7?).
set -u
REPO=/root/ptcg/repo
VOCAB=$REPO/data/cardfirst_b_v39.json
PAIRS=/root/dpo_r2b.jsonl.gz
FROM=/root/out/dpo_r1
OUT=/root/out/dpo_r2
STATE=/root/loop_dpo
MIRROR_SO=$REPO/data/kaggle_engine_ext/libcg_mirror.so
mkdir -p "$STATE"
cd "$REPO"
say() { echo "[dpo2 $(date -u +%m-%d_%H:%M:%S)] $*"; }

[ -f "$PAIRS" ] || { say "STOP: no $PAIRS"; exit 1; }
[ -f "$FROM/domain_embeddings.pt" ] || { say "STOP: $FROM is not a finished checkpoint"; exit 1; }

say "probe start (ref = dpo_r1)"
python3 tools/instance/dpo_teacher.py --data "$PAIRS" --init-from "$FROM" \
  --card-first "$VOCAB" --out /root/out/dpo_probe2 --probe --lr 5e-5 > "$STATE/probe2.log" 2>&1
grep -aE "PROBE" "$STATE/probe2.log" || { say "probe crashed -- see $STATE/probe2.log"; exit 1; }
grep -aq "PROBE OK" "$STATE/probe2.log" || { say "PROBE FAILED -- stopping before the round"; exit 1; }

say "train start"
python3 tools/instance/dpo_teacher.py --data "$PAIRS" --init-from "$FROM" \
  --card-first "$VOCAB" --out "$OUT" --epochs 3 --beta 0.1 --lr 5e-5 \
  > "$STATE/train2.log" 2>&1 || { say "train FAILED"; tail -5 "$STATE/train2.log"; exit 1; }
grep -aE "FINAL|saved" "$STATE/train2.log" | tail -3
[ -f "$OUT/domain_embeddings.pt" ] || { say "STOP: no domain_embeddings.pt in $OUT"; exit 1; }

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
      --out "$STATE/gate_dpo2.$j.json" > "$STATE/gate_dpo2.$j.log" 2>&1 &
  j=$((j+1))
done < "$STATE/shards.txt"
say "launched $j gate shards"
wait
python3 - <<'PY'
import json, math, statistics as st

def load_shards(prefix, n=3):
    d = {}
    for k in range(n):
        try:
            d.update(json.load(open("/root/loop_dpo/%s.%d.json" % (prefix, k)))["decks"])
        except Exception as e:
            print("shard %s.%d unreadable: %s" % (prefix, k, e))
    return d

cur = load_shards("gate_dpo2")
prev = load_shards("gate_dpo1")
base = json.load(open("/root/loop_stage1/gate_r1.json"))["decks"]

p = [v["p"] for v in cur.values()]
print("[gate] dpo_r2 | %d decks | mean %.1f%% | median %.1f%% | below50 %d"
      % (len(cur), 100*st.mean(p), 100*st.median(p), sum(1 for x in p if x < .5)))

def paired(name, ref):
    both = sorted(set(cur) & set(ref))
    if not both:
        print("[paired vs %s] no overlap" % name); return
    dd = [cur[k]["p"] - ref[k]["p"] for k in both]
    m = st.mean(dd); se = st.stdev(dd)/math.sqrt(len(dd)) if len(dd) > 1 else 0.0
    print("[paired vs %s, %d decks] %+.4f +- %.4f  t %+.2f  (up %d/%d)"
          % (name, len(both), m, se, m/se if se else 0, sum(1 for x in dd if x > 0), len(dd)))
    for k in sorted(both, key=lambda k: cur[k]["p"]-ref[k]["p"]):
        print("   %-22s %5.1f%% -> %5.1f%%  (%+.1f)"
              % (k, 100*ref[k]["p"], 100*cur[k]["p"], 100*(cur[k]["p"]-ref[k]["p"])))

paired("dpo_r1 (per-round)", prev)
paired("i2_r7 (cumulative)", base)
PY
say DPO_ROUND2_DONE
