#!/usr/bin/env bash
# DPO round 6. Same recipe as round 5 -- init from the ADOPTED checkpoint, beta's reference
# pinned at the SFT -- with the round-5 gate's failures fixed:
#
#   * the GPU is checked empty and the shards are staggered. Round 5 launched three scorers into
#     a card that still held the training process; two died inside resize_token_embeddings and
#     the verdict was computed from the one survivor and reported as an 11-deck result.
#   * a missing shard is a HARD FAILURE, not a printed "unreadable" line.
#   * mirror_match passes mean_resizing=False, so the load no longer allocates a Cholesky of the
#     embedding covariance whose output is overwritten four lines later.
#
# New in the DATA: slowking is in the opponent set for the first time (22 deck orderings vs 20),
# so the ladder's #1/#2 finally appears in both the pairs and the states.
set -u
REPO=/root/ptcg/repo
VOCAB=$REPO/data/cardfirst_b_v39.json
PAIRS=/root/dpo_r6.jsonl.gz
FROM=/root/out/dpo_r5          # adopted: 11 decks, +1.67pt +- 0.67 vs dpo_r4 (t +2.49)
REF=/root/out/i2_r7            # the SFT checkpoint -- NOT re-anchored per round, on purpose
OUT=/root/out/dpo_r6
STATE=/root/loop_dpo
MIRROR_SO=$REPO/data/kaggle_engine_ext/libcg_mirror.so
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
mkdir -p "$STATE"; cd "$REPO"
say() { echo "[dpo6 $(date -u +%m-%d_%H:%M:%S)] $*"; }

# ---------------------------------------------------------------- 0. wait for the pairs
# instance1 builds them (playouts are CPU work and it has 61 effective cores) and scps them
# here. Waiting rather than failing: the two-instance link drops silently often enough that an
# exit here would just mean a human restart later.
for _ in $(seq 1 90); do
    [ -s "$PAIRS" ] && break
    sleep 60
done
[ -s "$PAIRS" ] || { say "STOP: $PAIRS never arrived (waited 90 min)"; exit 1; }
say "pairs: $(zcat "$PAIRS" | wc -l)"

# ---------------------------------------------------------------- 1. probe
say "probe (unsmoothed on purpose: cDPO raises the achievable floor to ~H(eps))"
python3 tools/instance/dpo_teacher.py --data "$PAIRS" --init-from "$FROM" \
    --card-first "$VOCAB" --out /root/out/dpo_probe6 --probe --lr 5e-5 \
    > "$STATE/probe6.log" 2>&1
grep -aq "PROBE OK" "$STATE/probe6.log" || { say "PROBE FAILED"; tail -4 "$STATE/probe6.log"; exit 1; }

# ---------------------------------------------------------------- 2. train
say "train: init $FROM, reference pinned at $REF"
python3 tools/instance/dpo_teacher.py --data "$PAIRS" --init-from "$FROM" --ref-from "$REF" \
    --card-first "$VOCAB" --out "$OUT" --epochs 3 --beta 0.1 --lr 5e-5 --cdpo-calibrated \
    > "$STATE/train6.log" 2>&1 || { say "train FAILED"; tail -6 "$STATE/train6.log"; exit 1; }
grep -aE "\[ref\]|\[cdpo\]|FINAL|saved" "$STATE/train6.log" | tail -6
[ -f "$OUT/domain_embeddings.pt" ] || { say "STOP: no checkpoint in $OUT"; exit 1; }

# ---------------------------------------------------------------- 3. gate
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
for _ in $(seq 1 20); do
    [ "$used" -le 2000 ] && break
    sleep 30
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
done
[ "$used" -le 2000 ] || { say "STOP: GPU still holds ${used} MiB"; exit 1; }
say "GPU clear (${used} MiB)"

DECKS=$(python3 -c "import sys;sys.path.insert(0,'tools');import rl_config;print(','.join(rl_config.STAGE_C_TARGETS))")
python3 - "$DECKS" > "$STATE/shards.txt" <<'PYX'
import sys
d = sys.argv[1].split(",")
for i in range(3):
    print(" ".join("--deck " + x for x in d[i::3]))
PYX
say "gate: 11 decks x 229 vs engine_v2"
rm -f "$STATE"/gate_dpo6.*.json
j=0
while read -r DK; do
    [ -n "$DK" ] || continue
    PYTHONPATH=cg-lib nohup python3 tools/mirror_match.py $DK --a engine --b "qwen:$OUT" \
        --max-games 229 --mirror --seed 1 --mirror-so "$MIRROR_SO" \
        --out "$STATE/gate_dpo6.$j.json" > "$STATE/gate_dpo6.$j.log" 2>&1 &
    j=$((j+1))
    sleep 120     # the peak is at model load; three at once is what killed round 5
done < "$STATE/shards.txt"
say "launched $j gate shards"; wait
miss=0
for k in 0 1 2; do
    [ -s "$STATE/gate_dpo6.$k.json" ] || { say "SHARD $k PRODUCED NO JSON"; tail -15 "$STATE/gate_dpo6.$k.log"; miss=1; }
done
[ "$miss" -eq 0 ] || { say "STOP: incomplete gate -- refusing to decide adoption on a subset"; exit 1; }

# ---------------------------------------------------------------- 4. verdict
python3 - <<'PY'
import json, math, statistics as st
def load(p, n=3):
    d = {}
    for k in range(n):
        d.update(json.load(open("/root/loop_dpo/%s.%d.json" % (p, k)))["decks"])
    return d
cur = load("gate_dpo6"); r5 = load("gate_dpo5b")
base = json.load(open("/root/loop_stage1/gate_r1.json"))["decks"]
p = [v["p"] for v in cur.values()]
print("[gate] dpo_r6 | %d decks | mean %.1f%% | median %.1f%% | below50 %d"
      % (len(cur), 100*st.mean(p), 100*st.median(p), sum(1 for x in p if x < .5)))
def sp(v, s):
    w, l = v["seat%d" % s]; return w/(w+l) if (w+l) else float("nan")
res = {}
def paired(key, name, ref):
    ks = sorted(set(cur) & set(ref))
    if not ks: return
    dd = [cur[k]["p"] - ref[k]["p"] for k in ks]
    se = st.stdev(dd)/math.sqrt(len(dd))
    t = st.mean(dd)/se if se else 0.0
    res[key] = (100*st.mean(dd), 100*se, t, len(ks))
    print("[paired vs %s, %d decks] %+.4f +- %.4f  t %+.2f  (up %d/%d)"
          % (name, len(ks), st.mean(dd), se, t, sum(1 for x in dd if x > 0), len(dd)))
    for s in (0, 1):
        d2 = [sp(cur[k], s) - sp(ref[k], s) for k in ks]
        print("      seat%d %+.2fpt +- %.2f" % (s, 100*st.mean(d2),
              100*st.stdev(d2)/math.sqrt(len(d2))))
paired("r5", "dpo_r5 (the adopted checkpoint)", r5)
paired("sft", "i2_r7 (cumulative)", base)
for k in sorted(set(cur) & set(r5), key=lambda k: cur[k]["p"]-r5[k]["p"]):
    print("   %-22s r5 %5.1f%% -> r6 %5.1f%%  (%+.1f)"
          % (k, 100*r5[k]["p"], 100*cur[k]["p"], 100*(cur[k]["p"]-r5[k]["p"])))
d, se, t, n = res.get("r5", (0.0, 0.0, 0.0, 0))
worse = (d <= -2.0 and t <= -2.0)
adopt = "dpo_r5" if worse else "dpo_r6"
print("VERDICT delta_vs_r5 %+.2fpt +- %.2f (t %+.2f, %d decks) -> ADOPT %s"
      % (d, se, t, n, adopt))
open("/root/loop_dpo/adopt_r6.txt", "w").write(adopt + "\n")
PY
say DPO_ROUND6_DONE
