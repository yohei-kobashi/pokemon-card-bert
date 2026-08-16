#!/usr/bin/env bash
# DPO round 4. Three things are new and all of them were bugs before:
#   1. labels are no longer sign-inverted for the second seat (dpo_branch pilot_i fix)
#   2. the data is CROSS-DECK: dragapult_dusknoir against the ten Stage-C opponents, which is
#      the matchup shape the ladder actually has -- rounds 1-3 were same-deck mirrors
#   3. beta's reference stays anchored at the SFT checkpoint (--ref-from i2_r7) instead of
#      being re-anchored to the previous round, which is what let the policy walk away from
#      the SFT with nothing measuring the total distance
set -u
REPO=/root/ptcg/repo
VOCAB=$REPO/data/cardfirst_b_v39.json
PAIRS=/root/dpo_r4.jsonl.gz
FROM=/root/out/dpo_r1                 # adopted on the 11-deck gate: 52.0%
REF=/root/out/i2_r7                   # the SFT checkpoint
OUT=/root/out/dpo_r4
STATE=/root/loop_dpo
MIRROR_SO=$REPO/data/kaggle_engine_ext/libcg_mirror.so
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
mkdir -p "$STATE"; cd "$REPO"
say() { echo "[dpo4 $(date -u +%m-%d_%H:%M:%S)] $*"; }
[ -f "$PAIRS" ] || { say "STOP: no $PAIRS"; exit 1; }

say "probe (unsmoothed, on purpose: cDPO raises the achievable floor to ~H(eps))"
python3 tools/instance/dpo_teacher.py --data "$PAIRS" --init-from "$FROM" \
  --card-first "$VOCAB" --out /root/out/dpo_probe4 --probe --lr 5e-5 > "$STATE/probe4.log" 2>&1
grep -aq "PROBE OK" "$STATE/probe4.log" || { say "PROBE FAILED"; tail -3 "$STATE/probe4.log"; exit 1; }

say "train: init $FROM, reference pinned at $REF"
python3 tools/instance/dpo_teacher.py --data "$PAIRS" --init-from "$FROM" --ref-from "$REF" \
  --card-first "$VOCAB" --out "$OUT" --epochs 3 --beta 0.1 --lr 5e-5 --cdpo-calibrated \
  > "$STATE/train4.log" 2>&1 || { say "train FAILED"; tail -6 "$STATE/train4.log"; exit 1; }
grep -aE "\[ref\]|\[cdpo\]|FINAL|saved" "$STATE/train4.log" | tail -6
[ -f "$OUT/domain_embeddings.pt" ] || { say "STOP: no checkpoint in $OUT"; exit 1; }

say "gate: 11 decks x 229 vs engine_v2 -- the user's adoption criterion"
DECKS=$(python3 -c "import sys;sys.path.insert(0,'tools');import rl_config;print(','.join(rl_config.STAGE_C_TARGETS))")
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
      --out "$STATE/gate_dpo4.$j.json" > "$STATE/gate_dpo4.$j.log" 2>&1 &
  j=$((j+1))
done < "$STATE/shards.txt"
say "launched $j gate shards"; wait
python3 - <<'PY'
import json, math, statistics as st
def load(p, n=3):
    d = {}
    for k in range(n):
        try: d.update(json.load(open("/root/loop_dpo/%s.%d.json" % (p, k)))["decks"])
        except Exception as e: print("shard %s.%d unreadable: %s" % (p, k, e))
    return d
cur = load("gate_dpo4"); r1 = load("gate_dpo1")
base = json.load(open("/root/loop_stage1/gate_r1.json"))["decks"]
p = [v["p"] for v in cur.values()]
print("[gate] dpo_r4 | %d decks | mean %.1f%% | median %.1f%% | below50 %d"
      % (len(cur), 100*st.mean(p), 100*st.median(p), sum(1 for x in p if x < .5)))
def sp(v, s):
    w, l = v["seat%d" % s]; return w/(w+l) if (w+l) else float("nan")
def paired(name, ref):
    ks = sorted(set(cur) & set(ref))
    if not ks: return
    dd = [cur[k]["p"] - ref[k]["p"] for k in ks]
    se = st.stdev(dd)/math.sqrt(len(dd))
    print("[paired vs %s, %d decks] %+.4f +- %.4f  t %+.2f  (up %d/%d)"
          % (name, len(ks), st.mean(dd), se, st.mean(dd)/se if se else 0,
             sum(1 for x in dd if x > 0), len(dd)))
    for s in (0, 1):
        d2 = [sp(cur[k], s) - sp(ref[k], s) for k in ks]
        print("      seat%d %+.2fpt +- %.2f" % (s, 100*st.mean(d2),
              100*st.stdev(d2)/math.sqrt(len(d2))))
paired("dpo_r1 (the adopted checkpoint)", r1)
paired("i2_r7 (cumulative)", base)
for k in sorted(set(cur) & set(base), key=lambda k: cur[k]["p"]-base[k]["p"]):
    print("   %-22s %5.1f%% -> %5.1f%%  (%+.1f)"
          % (k, 100*base[k]["p"], 100*cur[k]["p"], 100*(cur[k]["p"]-base[k]["p"])))
PY
say DPO_ROUND4_DONE
