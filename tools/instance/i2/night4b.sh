#!/usr/bin/env bash
# instance2, overnight: does the pair-confidence filter help the 4B, as it did the encoder?
#
# WHAT TONIGHT'S INVESTIGATION ACTUALLY PRODUCED. The ogerpon dig ended with four refuted
# hypotheses and one transferable result, and the transferable one is not about ogerpon at all:
# on instance1, training the cross-encoder on EVERY branched pair moved held-out conformance
# 54.3 -> 53.6 (down), while keeping only |qw-ql| >= 0.35 moved it 52.1 -> 58.1. With 24
# playouts the Q estimate's SE is ~0.2 and the median margin is 0.26, so the low-confidence
# majority outvotes the informative minority. `dpo_teacher.py` -- the 4B trainer -- had NO such
# filter. That is the experiment worth a night.
#
# WHY NOT MORE OGERPON. One round there returned +1.00 +- 2.00 off 397 pairs, and those pairs
# are unusually flat: only 84 survive qmin 0.25 and 49 survive 0.35. A second round would buy
# another underpowered null. Collecting across the whole field gives both a bigger pair set and
# a gate with cells to detect regressions in.
#
# DESIGN. One collection, ONE pair set, TWO trainings that differ only in the filter, and a
# paired gate with the unfiltered arm as baseline -- so the number produced is exactly "what the
# filter is worth", not "what another round is worth".
#
# OPPONENTS ARE engine_v2, DELIBERATELY. A 4B-vs-4B game costs ~34 s against ~17 s with the
# heuristic on the far side, and tonight the binding constraint is wall clock, not opponent
# realism: 4B opponents would halve every sample size in a test that is already about
# statistical power. engine_v2 is also the yardstick instance1 gates on, so the numbers are
# comparable across the two machines.
set -u
REPO=/root/ptcg/repo; cd "$REPO"
LOG=${LOG:-/root/night4b.log}
VOCAB=$REPO/data/cardfirst_b_v39.json
REF=/root/out/i2_r7
PREV=${PREV:-/root/out/dpo_r8}
SO=$REPO/data/kaggle_engine_ext/libcg_mirror.so
OPPS=${OPPS:-marnie_grimmsnarl,alakazam_nz,dragapult,dudunsparce_box,archaludon,ogerpon_mono,mega_abomasnow_sample,ethan_hooh}
GAMES=${GAMES:-50}            # per deck per shard; 8 decks x 50 x 2 shards = 800 games
GATE_GAMES=${GATE_GAMES:-50}
TAG=${TAG:-night4b}
PAIRS=/root/pairs_$TAG.jsonl.gz
TR_GLOB=/root/traces_$TAG.s
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=cg-lib
say() { echo "[n4b $(date -u +%m-%d_%H:%M:%S)] $*" >> "$LOG"; }
DEADLINE=$(date -u -d "+${HOURS:-7} hours" +%s)
left() { echo $(( (DEADLINE - $(date -u +%s)) / 60 )); }

gpu_wait() {
    for _ in $(seq 1 60); do
        u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        [ "$u" -le 2000 ] && return 0
        sleep 30
    done
    say "GPU still busy (${u} MiB) -- continuing anyway"; return 0
}

say "start; $(left) min of budget. policy $PREV, opponents engine_v2, decks $OPPS"

# ------------------------------------------------------------------ 1. collect
if ! { [ -s ${TR_GLOB}0.jsonl.gz ] && [ -s ${TR_GLOB}1.jsonl.gz ]; }; then
    rm -f ${TR_GLOB}*.jsonl.gz
    gpu_wait
    say "collect: 2 shards x 8 decks x $GAMES games"
    for SH in 0 1; do
        nohup python3 tools/lm_mirror_log.py --model "qwen:$PREV" --deck-model engine \
            --fmt prompt --protagonist dragapult_dusknoir --decks "$OPPS" --games "$GAMES" \
            --seed $((800000 + SH * 1000)) \
            --out /root/lmlog_$TAG.s$SH.jsonl.gz --trace-out ${TR_GLOB}$SH.jsonl.gz \
            --mirror-so "$SO" > /root/collect_$TAG.s$SH.log 2>&1 &
    done
    wait
fi
for SH in 0 1; do
    [ -s ${TR_GLOB}$SH.jsonl.gz ] || { say "STOP: shard $SH empty"; exit 1; }
done
say "collected; $(left) min left"

# ------------------------------------------------------------------ 2. branch on instance1
if [ "$(zcat "$PAIRS" 2>/dev/null | head -1 | wc -l)" -eq 0 ]; then
    rm -f "$PAIRS"
    echo "$TAG|dragapult_dusknoir|${BR_BUDGET:-12000}|${BR_PLAYOUTS:-24}|${BR_PERGAME:-15}" > /root/branch_request2
    say "branch requested from instance1"
    for _ in $(seq 1 ${BR_WAIT:-100}); do [ -s "$PAIRS" ] && break; sleep 60; done
    for _ in $(seq 1 60); do
        a=$(stat -c %s "$PAIRS" 2>/dev/null || echo 0); sleep 10
        b=$(stat -c %s "$PAIRS" 2>/dev/null || echo 0)
        [ "$a" = "$b" ] && [ "$a" != 0 ] && gzip -t "$PAIRS" 2>/dev/null && break
    done
    if [ ! -s "$PAIRS" ]; then
        [ "${BR_FALLBACK:-1}" = 1 ] || { say "STOP: no pairs from instance1 in ${BR_WAIT:-100} min (fallback disabled)"; exit 1; }
        say "FALLBACK: local branch, budget 4000 playouts 16 -- smaller and noisier"
        rm -f /root/branch_request2
        RL_PRIZE_GAMMA=0.25 CUDA_VISIBLE_DEVICES= nice -n 5 python3 tools/dpo_branch.py \
            --traces "$(ls ${TR_GLOB}*.jsonl.gz | paste -sd,)" --only-deck dragapult_dusknoir \
            --budget 4000 --per-game 15 --margin-min 0.01 --playouts 16 --workers 12 \
            --seed 45000 --out "$PAIRS" || { say "local branch FAILED"; exit 1; }
    fi
fi
NP=$(zcat "$PAIRS" | wc -l)
say "pairs: $NP; $(left) min left"
[ "$NP" -ge "${MINROWS:-500}" ] || { say "STOP: only $NP pairs"; exit 1; }

# ------------------------------------------------------------------ 3. pick the threshold
# Chosen from THIS pair set rather than copied from instance1: the surviving fraction is what
# matters (34% there), and a threshold that leaves 40 rows would test the filter's arithmetic
# instead of its effect.
QMIN=$(python3 - "$PAIRS" "${MINROWS:-500}" <<'PY'
import gzip, json, sys
g = sorted(abs(float(json.loads(l).get("qw", 0)) - float(json.loads(l).get("ql", 0)))
           for l in gzip.open(sys.argv[1], "rt"))
n = len(g)
for q in (0.50, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15):
    keep = sum(1 for x in g if x >= q)
    if keep >= max(int(sys.argv[2]), int(0.30 * n)):
        print("%.2f" % q); break
else:
    print("0.15")
PY
)
[ -n "$QMIN" ] || { say "STOP: qmin selection produced nothing (is $PAIRS a complete gzip?)"; exit 1; }
say "qmin chosen: $QMIN"

# ------------------------------------------------------------------ 4. two trainings, one difference
for ARM in base filt; do
    Q=0.0; [ "$ARM" = "filt" ] && Q=$QMIN
    OUT=/root/out/lora_${TAG}_$ARM
    [ -f "$OUT/domain_embeddings.pt" ] && { say "$ARM already trained"; continue; }
    gpu_wait
    say "train $ARM (qmin $Q)"
    python3 tools/instance/dpo_teacher.py --data "$PAIRS" --init-from "$PREV" --ref-from "$REF" \
        --card-first "$VOCAB" --out "$OUT" --epochs "${EPOCHS:-3}" --beta 0.1 --lr 5e-5 --cdpo-calibrated \
        --qmin "$Q" > /root/train_${TAG}_$ARM.log 2>&1 \
        || { say "$ARM train FAILED"; tail -5 /root/train_${TAG}_$ARM.log >> "$LOG"; exit 1; }
    grep -aE "^\[data\]|FINAL|saved" /root/train_${TAG}_$ARM.log | tail -3 >> "$LOG"
done
say "trained; $(left) min left"

# ------------------------------------------------------------------ 5. gate, filt vs base
gpu_wait
say "gate: base (baseline) vs filt, 8 opponents x $GATE_GAMES games"
python3 -u tools/gate_protagonist.py --deck dragapult_dusknoir --opp "$OPPS" \
    --games "$GATE_GAMES" --seed 99000 --baseline base --opp-spec engine \
    --arm "base=qwen:/root/out/lora_${TAG}_base" \
    --arm "filt=qwen:/root/out/lora_${TAG}_filt" \
    --mirror-so "$SO" --out /root/gate_$TAG.json > /root/gate_$TAG.log 2>&1 \
    || { say "gate FAILED"; tail -8 /root/gate_$TAG.log >> "$LOG"; exit 1; }
grep -aE "vs |^base|^filt|arm " /root/gate_$TAG.log | tail -14 >> "$LOG"

python3 - /root/gate_$TAG.json "$QMIN" <<'PY' >> "$LOG" 2>&1
import json, sys
j = json.load(open(sys.argv[1]))
b, f = j["arms"]["base"], j["arms"]["filt"]
print("\n=============== does the pair filter help the 4B? ===============")
print("qmin                 %s" % sys.argv[2])
print("unfiltered (today)   %.1f%%" % b["win_rate"])
print("filtered             %.1f%%   delta %+.2f +- %.2f (t %+.2f)"
      % (f["win_rate"], f["delta_vs_baseline"], f["se"],
         f["delta_vs_baseline"] / f["se"] if f["se"] else 0.0))
print("\nper opponent:")
for o in sorted({k.split("|", 1)[1] for k in j["cells"]}):
    cb = j["cells"].get("base|" + o); cf = j["cells"].get("filt|" + o)
    if cb and cf:
        print("  %-24s %5.1f%% -> %5.1f%%  %+5.1f"
              % (o, 100.0 * cb["win"] / cb["games"], 100.0 * cf["win"] / cf["games"],
                 100.0 * (cf["win"] / cf["games"] - cb["win"] / cb["games"])))
print("\nRead with instance1's encoder result: there, filtering turned a held-out conformance")
print("LOSS into a 6-point gain. If the 4B agrees, the filter is a property of the DATA and")
print("instance1's qmin change is corroborated on a second model and a second trainer.")
PY
say "NIGHT4B_DONE; $(left) min of budget remained"
