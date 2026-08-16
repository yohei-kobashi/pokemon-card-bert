#!/usr/bin/env bash
# One complete DPO round, driven ENTIRELY from instance2:
#
#   collect (GPU) -> branch (instance1 via branchd; local fallback) -> probe -> train (GPU)
#   -> gate (GPU) -> verdict -> adopt_rN.txt
#
# Rounds 1-6 needed a human to carry traces to instance1, start the branch, and carry pairs
# back; round 6 sat 4 hours in that gap and its ship then failed on the link. The transfers are
# now instance1's job (branchd polls; instance2 cannot reach instance1 -- the vast proxy
# authenticates against account keys, so authorized_keys on instance1 is never consulted), and
# the ONLY thing this script does about it is write a request file and wait.
#
#   bash /root/round.sh 7
set -u
N=${1:?usage: round.sh <round number>}
# The user ended the fleet-wide loop after the round in flight (2026-08-10): the plan is now
# per-deck LoRAs on the live top-5, not more 11-deck rounds. rounds.sh re-invokes this script
# per round, so the flag is read at the START of each round and only stops the NEXT one.
LAST_ROUND=$(cat /root/STOP_ROUNDS 2>/dev/null || echo "")
if [ -n "$LAST_ROUND" ] && [ "$N" -gt "$LAST_ROUND" ]; then
    echo "STOP_ROUNDS=$LAST_ROUND -- not starting round $N"
    exit 1
fi
PREV=${PREV:-}
REPO=/root/ptcg/repo
VOCAB=$REPO/data/cardfirst_b_v39.json
REF=/root/out/i2_r7                 # beta's anchor: the SFT, never re-anchored per round
STATE=/root/loop_dpo
SO=$REPO/data/kaggle_engine_ext/libcg_mirror.so
GAMES=${GAMES:-150}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
mkdir -p "$STATE"; cd "$REPO"
say() { echo "[r$N $(date -u +%m-%d_%H:%M:%S)] $*"; }

if [ -z "$PREV" ]; then
    PREV=$(cat "$STATE/adopt_r$((N-1)).txt" 2>/dev/null || true)
    [ -n "$PREV" ] || { say "STOP: no adopt_r$((N-1)).txt; pass PREV=/root/out/dpo_rX"; exit 1; }
    PREV=/root/out/$PREV
fi
[ -d "$PREV" ] || { say "STOP: no checkpoint at $PREV"; exit 1; }
OUT=/root/out/dpo_r$N
PAIRS=/root/dpo_r$N.jsonl.gz
say "starting from $PREV"

FREE_G=$(df -BG /root | awk 'NR==2{gsub("G","",$4); print $4}')
[ "$FREE_G" -ge 15 ] || { say "STOP: only ${FREE_G}G free"; exit 1; }

gpu_wait() {
    local u
    for _ in $(seq 1 40); do
        u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
        [ "$u" -le 2000 ] && { say "GPU clear (${u} MiB)"; return 0; }
        sleep 30
    done
    say "STOP: GPU still holds ${u} MiB after 20 min"; exit 1
}

# ---------------------------------------------------------------- 1. collect
if ! ls /root/traces_r$N.s0.jsonl.gz >/dev/null 2>&1; then
    gpu_wait
    OPPS=$(python3 -c "
import sys; sys.path.insert(0,'tools'); import rl_config
d=[x for x in rl_config.STAGE_C_TARGETS if x!='dragapult_dusknoir']
d.append('slowking')
print(','.join(d))")
    say "collect: dragapult_dusknoir vs $OPPS (model $PREV)"
    j=0
    for SH in 0 1 2; do
        DK=$(python3 -c "print(','.join('$OPPS'.split(',')[$SH::3]))")
        [ -n "$DK" ] || continue
        PYTHONPATH=cg-lib nohup python3 tools/lm_mirror_log.py --model "qwen:$PREV" \
            --protagonist dragapult_dusknoir --decks "$DK" --games "$GAMES" \
            --seed $((100000 + N * 10000 + SH * 1000)) \
            --out /root/lmlog_r$N.s$SH.jsonl.gz --trace-out /root/traces_r$N.s$SH.jsonl.gz \
            --mirror-so "$SO" > /root/collect_r$N.s$SH.log 2>&1 &
        j=$((j+1))
    done
    say "launched $j collection shards"; wait
fi
ls /root/traces_r$N.s*.jsonl.gz >/dev/null 2>&1 || { say "STOP: no traces"; exit 1; }

# ---------------------------------------------------------------- 2. branch via instance1
if [ ! -s "$PAIRS" ]; then
    echo "$N" > /root/branch_request
    say "branch requested from instance1 (20k branch points, 32 playouts); waiting"
    for _ in $(seq 1 90); do
        [ -s "$PAIRS" ] && break
        sleep 60
    done
    if [ ! -s "$PAIRS" ]; then
        # LOUD fallback, never silent: a fifth of the budget at half the playouts is what
        # 13.44 effective cores can deliver in about an hour. The verdict readers must know.
        say "FALLBACK: instance1 did not deliver in 90 min -- building LOCALLY at budget 4000,"
        say "FALLBACK: playouts 16. This round's pair set is SMALLER AND NOISIER than r4-r6."
        rm -f /root/branch_request
        CUDA_VISIBLE_DEVICES= PYTHONPATH=cg-lib nice -n 5 python3 tools/dpo_branch.py \
            --traces "$(ls /root/traces_r$N.s*.jsonl.gz | paste -sd,)" \
            --budget 4000 --per-game 15 --margin-min 0.01 --playouts 16 --workers 12 \
            --out "$PAIRS" || { say "local branch FAILED too"; exit 1; }
    fi
fi
say "pairs: $(zcat "$PAIRS" | wc -l)"

# ---------------------------------------------------------------- 3. probe + train
gpu_wait
say "probe"
python3 tools/instance/dpo_teacher.py --data "$PAIRS" --init-from "$PREV" \
    --card-first "$VOCAB" --out /root/out/dpo_probe$N --probe --lr 5e-5 \
    > "$STATE/probe$N.log" 2>&1
grep -aq "PROBE OK" "$STATE/probe$N.log" || { say "PROBE FAILED"; tail -4 "$STATE/probe$N.log"; exit 1; }
say "train: init $PREV, reference pinned at $REF"
python3 tools/instance/dpo_teacher.py --data "$PAIRS" --init-from "$PREV" --ref-from "$REF" \
    --card-first "$VOCAB" --out "$OUT" --epochs 3 --beta 0.1 --lr 5e-5 --cdpo-calibrated \
    > "$STATE/train$N.log" 2>&1 || { say "train FAILED"; tail -6 "$STATE/train$N.log"; exit 1; }
grep -aE "\[ref\]|FINAL|saved" "$STATE/train$N.log" | tail -4
[ -f "$OUT/domain_embeddings.pt" ] || { say "STOP: no checkpoint in $OUT"; exit 1; }

# ---------------------------------------------------------------- 4. gate
gpu_wait
DECKS=$(python3 -c "import sys;sys.path.insert(0,'tools');import rl_config;print(','.join(rl_config.STAGE_C_TARGETS))")
python3 - "$DECKS" > "$STATE/shards.txt" <<'PYX'
import sys
d = sys.argv[1].split(",")
for i in range(3):
    print(" ".join("--deck " + x for x in d[i::3]))
PYX
say "gate: 11 decks x 229 vs engine_v2"
rm -f "$STATE"/gate_dpo$N.*.json
j=0
while read -r DK; do
    [ -n "$DK" ] || continue
    PYTHONPATH=cg-lib nohup python3 tools/mirror_match.py $DK --a engine --b "qwen:$OUT" \
        --max-games 229 --mirror --seed 1 --mirror-so "$SO" \
        --out "$STATE/gate_dpo$N.$j.json" > "$STATE/gate_dpo$N.$j.log" 2>&1 &
    j=$((j+1)); sleep 120
done < "$STATE/shards.txt"
say "launched $j gate shards"; wait
for k in 0 1 2; do
    [ -s "$STATE/gate_dpo$N.$k.json" ] || { say "SHARD $k NO JSON -- refusing a subset verdict"; exit 1; }
done

# ---------------------------------------------------------------- 5. verdict
PREVNAME=$(basename "$PREV")
python3 - "$N" "$PREVNAME" <<'PY'
import json, math, statistics as st, sys, glob
N, prevname = sys.argv[1], sys.argv[2]
def load(pat):
    d = {}
    for p in sorted(glob.glob(pat)):
        d.update(json.load(open(p))["decks"])
    return d
cur = load("/root/loop_dpo/gate_dpo%s.*.json" % N)
# The previous gate: dpo_r6 -> gate_dpo6 (round 5's honest 11-deck rerun is gate_dpo5b).
tag = prevname.replace("dpo_r", "dpo")
prev = load("/root/loop_dpo/gate_%sb.*.json" % tag) or load("/root/loop_dpo/gate_%s.*.json" % tag)
base = json.load(open("/root/loop_stage1/gate_r1.json"))["decks"]
p = [v["p"] for v in cur.values()]
print("[gate] dpo_r%s | %d decks | mean %.1f%% | median %.1f%% | below50 %d"
      % (N, len(cur), 100*st.mean(p), 100*st.median(p), sum(1 for x in p if x < .5)))
res = {}
def paired(key, name, ref):
    ks = sorted(set(cur) & set(ref))
    if not ks:
        print("[paired vs %s] NO OVERLAP" % name); return
    dd = [cur[k]["p"] - ref[k]["p"] for k in ks]
    se = st.stdev(dd)/math.sqrt(len(dd)); t = st.mean(dd)/se if se else 0.0
    res[key] = (100*st.mean(dd), 100*se, t, len(ks))
    print("[paired vs %s, %d decks] %+.4f +- %.4f  t %+.2f  (up %d/%d)"
          % (name, len(ks), st.mean(dd), se, t, sum(1 for x in dd if x > 0), len(dd)))
paired("prev", prevname, prev)
paired("sft", "i2_r7 (cumulative)", base)
for k in sorted(set(cur) & set(prev), key=lambda k: cur[k]["p"]-prev[k]["p"]):
    print("   %-22s %5.1f%% -> %5.1f%%  (%+.1f)"
          % (k, 100*prev[k]["p"], 100*cur[k]["p"], 100*(cur[k]["p"]-prev[k]["p"])))
d, se, t, n = res.get("prev", (0.0, 0.0, 0.0, 0))
adopt = prevname if (d <= -2.0 and t <= -2.0) else ("dpo_r%s" % N)
print("VERDICT delta %+.2fpt +- %.2f (t %+.2f, %d decks) -> ADOPT %s" % (d, se, t, n, adopt))
open("/root/loop_dpo/adopt_r%s.txt" % N, "w").write(adopt + "\n")
PY
say "ROUND_${N}_DONE"
