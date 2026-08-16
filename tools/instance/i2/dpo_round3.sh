#!/usr/bin/env bash
# DPO round 3. ONE variable vs round 2: label quality. Same traces, same seat-fair points,
# same init (dpo_r1, the best checkpoint -- round 2 trained on noise and is discarded), but
# 64 playouts behind each label and a per-pair cDPO epsilon from the measured agreement curve.
# The PROBE deliberately runs WITHOUT --cdpo-calibrated: smoothing raises the achievable loss
# floor to ~H(eps) ~ 0.37, which would fail the < 0.15 PROBE OK threshold for a reason that
# has nothing to do with the optimizer it is there to test.
set -u
REPO=/root/ptcg/repo
VOCAB=$REPO/data/cardfirst_b_v39.json
PAIRS=/root/dpo_r3.jsonl.gz
FROM=/root/out/dpo_r1
OUT=/root/out/dpo_r3
STATE=/root/loop_dpo
MIRROR_SO=$REPO/data/kaggle_engine_ext/libcg_mirror.so
mkdir -p "$STATE"; cd "$REPO"
say() { echo "[dpo3 $(date -u +%m-%d_%H:%M:%S)] $*"; }

[ -f "$PAIRS" ] || { say "STOP: no $PAIRS"; exit 1; }
[ -f "$FROM/domain_embeddings.pt" ] || { say "STOP: $FROM is not a finished checkpoint"; exit 1; }

say "probe start (unsmoothed, on purpose)"
python3 tools/instance/dpo_teacher.py --data "$PAIRS" --init-from "$FROM" \
  --card-first "$VOCAB" --out /root/out/dpo_probe3 --probe --lr 5e-5 > "$STATE/probe3.log" 2>&1
grep -aE "PROBE" "$STATE/probe3.log" || { say "probe crashed"; tail -20 "$STATE/probe3.log"; exit 1; }
grep -aq "PROBE OK" "$STATE/probe3.log" || { say "PROBE FAILED"; exit 1; }

say "train start (--cdpo-calibrated)"
python3 tools/instance/dpo_teacher.py --data "$PAIRS" --init-from "$FROM" \
  --card-first "$VOCAB" --out "$OUT" --epochs 3 --beta 0.1 --lr 5e-5 --cdpo-calibrated \
  > "$STATE/train3.log" 2>&1 || { say "train FAILED"; tail -8 "$STATE/train3.log"; exit 1; }
grep -aE "\[cdpo\]|\[data\]|FINAL|saved" "$STATE/train3.log" | tail -5
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
      --out "$STATE/gate_dpo3.$j.json" > "$STATE/gate_dpo3.$j.log" 2>&1 &
  j=$((j+1))
done < "$STATE/shards.txt"
say "launched $j gate shards"
wait
python3 - <<'PY'
import json, math, statistics as st
def load(prefix, n=3):
    d = {}
    for k in range(n):
        try: d.update(json.load(open("/root/loop_dpo/%s.%d.json" % (prefix, k)))["decks"])
        except Exception as e: print("shard %s.%d unreadable: %s" % (prefix, k, e))
    return d
cur  = load("gate_dpo3")
prev = load("gate_dpo1")                                     # dpo_r1 = this round's init
base = json.load(open("/root/loop_stage1/gate_r1.json"))["decks"]
r2   = load("gate_dpo2")                                     # the noisy-label round
p = [v["p"] for v in cur.values()]
print("[gate] dpo_r3 | %d decks | mean %.1f%% | median %.1f%% | below50 %d"
      % (len(cur), 100*st.mean(p), 100*st.median(p), sum(1 for x in p if x < .5)))
def sp(v, s):
    w, l = v["seat%d" % s]; return w / (w + l) if (w + l) else float("nan")
def paired(name, ref, per_seat=False):
    ks = sorted(set(cur) & set(ref))
    if not ks: return
    dd = [cur[k]["p"] - ref[k]["p"] for k in ks]
    se = st.stdev(dd)/math.sqrt(len(dd)) if len(dd) > 1 else 0.0
    print("[paired vs %s, %d decks] %+.4f +- %.4f  t %+.2f  (up %d/%d)"
          % (name, len(ks), st.mean(dd), se, st.mean(dd)/se if se else 0,
             sum(1 for x in dd if x > 0), len(dd)))
    if per_seat:
        for s in (0, 1):
            d2 = [sp(cur[k], s) - sp(ref[k], s) for k in ks]
            e2 = st.stdev(d2)/math.sqrt(len(d2))
            print("      seat%d %+.2fpt +- %.2f" % (s, 100*st.mean(d2), 100*e2))
paired("dpo_r1 (per-round; the label-noise test)", prev, True)
paired("i2_r7 (cumulative)", base, True)
paired("dpo_r2 (the noisy-label round)", r2)
ks = sorted(set(cur) & set(base))
for k in sorted(ks, key=lambda k: cur[k]["p"]-base[k]["p"]):
    print("   %-22s i2_r7 %5.1f%% -> %5.1f%%  (%+.1f)"
          % (k, 100*base[k]["p"], 100*cur[k]["p"], 100*(cur[k]["p"]-base[k]["p"])))
PY
say DPO_ROUND3_DONE
