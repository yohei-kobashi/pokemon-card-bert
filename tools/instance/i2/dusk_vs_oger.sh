#!/usr/bin/env bash
# Can a 4B learn to beat ogerpon_mono, when the cross-encoder cannot?
#
# WHY THIS RUN EXISTS. dragapult_dusknoir wins 3-7% against ogerpon_mono across nine gate rounds
# while every other opponent sits at 19-53%, and an audit found no piloting error to blame: the
# Crushing Hammer rate is already DOUBLE its rate elsewhere, Phantom Dive's bench counters do
# reach their Tera bodies, and our own attachments contribute only 32 of their 154 damage. What
# the audit did find looks structural -- their four Crushing Hammers keep our Active at 1.05
# energy, Phantom Dive costs two, so the deck falls back on 70-damage Jet Headbutts into 210 HP
# bodies (44 uses vs 14 Phantom Dives; against marnie the ratio is the other way round).
#
# "Structural" is a claim about the DECK, and the way to test it is to change the PILOT as far as
# it can be changed. A 4B with 27x the parameters of the shipped cross-encoder, trained by the
# same DPO machinery on this one matchup, is that test:
#
#   it improves  -> the matchup is winnable and instance1's cross-encoder has something to learn
#   it does not  -> the matchup is the decklist, and instance1 should stop spending 46% of its
#                   data allocation on it (tools/field_alloc.py currently sends the most there)
#
# DIFFERENCE FROM deck_lora2.sh, which this is otherwise modelled on: BOTH seats are Qwen 4B
# here, where that script pairs a Qwen against the DeBERTa. That doubles the VRAM per game
# process (~17 GiB, not ~12), so this runs TWO collection shards rather than three -- three
# would want ~51 GiB on a 47.4 GiB card. It also means one --fmt serves both seats, since both
# are PROMPT_FMT; deck_lora2 needs the registry precisely because its two seats disagree.
set -u
REPO=/root/ptcg/repo
VOCAB=$REPO/data/cardfirst_b_v39.json
REF=/root/out/i2_r7                    # DPO anchor: the SFT, never re-anchored
PREV=${PREV:-/root/out/dpo_r8}         # current 4B dusknoir (adopted, 11-deck mean 59.0%)
OPP=${OPP:-ogerpon_mono}
N=${N:-1}
TAG=dusk_vs_${OPP}_r$N
STATE=/root/loop_deck
SO=$REPO/data/kaggle_engine_ext/libcg_mirror.so
GAMES=${GAMES:-120}                    # per shard, 2 shards
BUDGET=${BUDGET:-12000}
PER_GAME=${PER_GAME:-15}
PLAYOUTS=${PLAYOUTS:-24}
GATE_GAMES=${GATE_GAMES:-200}
OUT=/root/out/lora_$TAG
TR_GLOB=/root/traces_$TAG.s
PAIRS=/root/pairs_$TAG.jsonl.gz
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
mkdir -p "$STATE"; cd "$REPO"
say() { echo "[$TAG $(date -u +%m-%d_%H:%M:%S)] $*"; }

[ -d "$PREV" ] || { say "STOP: no checkpoint at $PREV"; exit 1; }
say "protagonist dragapult_dusknoir = qwen:$PREV | opponent $OPP = reg"

gpu_wait() {
    local u
    for _ in $(seq 1 60); do
        u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        [ "$u" -le 2000 ] && { say "GPU clear (${u} MiB)"; return 0; }
        sleep 30
    done
    say "STOP: GPU still holds ${u} MiB after 30 min"; exit 1
}

# Wait out whatever else owns the card first -- pass5's last deck, usually.
while pgrep -f "deck_lora2.sh" > /dev/null; do
    say "waiting for deck_lora2 to finish"; sleep 120
done

# ---------------------------------------------------------------- 0. the three baselines
# One number is not enough to read the result. These separate "the matchup is hard" from "the 4B
# OPPONENT is hard", and give instance1 a directly comparable figure: its own cross-encoder
# scores 13.3% against the engine_v2 ogerpon over 30 games.
if [ ! -s "$STATE/base_$TAG.json" ]; then
    gpu_wait
    say "baseline A: 4B dusknoir vs engine_v2 $OPP"
    python3 -u tools/gate_protagonist.py --deck dragapult_dusknoir --opp "$OPP" \
        --games 100 --seed 61000 --baseline cur --opp-spec engine \
        --arm "cur=qwen:$PREV" --mirror-so "$SO" \
        --out "$STATE/baseA_$TAG.json" > "$STATE/baseA_$TAG.log" 2>&1 \
        && grep -aE "^cur|vs " "$STATE/baseA_$TAG.log" | tail -3
    say "baseline B: 4B dusknoir vs 4B $OPP"
    python3 -u tools/gate_protagonist.py --deck dragapult_dusknoir --opp "$OPP" \
        --games 100 --seed 61000 --baseline cur --opp-spec reg \
        --arm "cur=qwen:$PREV" --mirror-so "$SO" \
        --out "$STATE/base_$TAG.json" > "$STATE/base_$TAG.log" 2>&1 \
        && grep -aE "^cur|vs " "$STATE/base_$TAG.log" | tail -3
fi

# ---------------------------------------------------------------- 1. collect
if ! { [ -s ${TR_GLOB}0.jsonl.gz ] && [ -s ${TR_GLOB}1.jsonl.gz ]; }; then
    rm -f ${TR_GLOB}*.jsonl.gz
    gpu_wait
    say "collect: dusknoir(qwen:$(basename $PREV)) vs $OPP(reg), 2 x $GAMES games"
    for SH in 0 1; do
        PYTHONPATH=cg-lib nohup python3 tools/lm_mirror_log.py \
            --model "qwen:$PREV" --deck-model reg --fmt prompt \
            --protagonist dragapult_dusknoir --decks "$OPP" --games "$GAMES" \
            --seed $((700000 + N * 10000 + SH * 1000)) \
            --out /root/lmlog_$TAG.s$SH.jsonl.gz --trace-out ${TR_GLOB}$SH.jsonl.gz \
            --mirror-so "$SO" > /root/collect_$TAG.s$SH.log 2>&1 &
    done
    say "launched 2 collection shards"; wait
fi
for SH in 0 1; do
    [ -s ${TR_GLOB}$SH.jsonl.gz ] || { say "STOP: shard $SH empty, see /root/collect_$TAG.s$SH.log"; exit 1; }
done

# ---------------------------------------------------------------- 2. branch on instance1
# --only-deck is OUR side: we are training the dusknoir pilot, so only its decisions are
# branched. instance1's branchd2 is tag-generic, so no change is needed over there.
if [ "$(zcat "$PAIRS" 2>/dev/null | head -1 | wc -l)" -eq 0 ]; then
    rm -f "$PAIRS"
    echo "$TAG|dragapult_dusknoir|$BUDGET|$PLAYOUTS|$PER_GAME" > /root/branch_request2
    say "branch requested from instance1 (tag $TAG)"
    for _ in $(seq 1 150); do [ -s "$PAIRS" ] && break; sleep 60; done
    if [ ! -s "$PAIRS" ]; then
        say "FALLBACK: building locally at budget 4000/playouts 16 -- SMALLER AND NOISIER"
        rm -f /root/branch_request2
        RL_PRIZE_GAMMA=0.25 CUDA_VISIBLE_DEVICES= PYTHONPATH=cg-lib nice -n 5 \
            python3 tools/dpo_branch.py --traces "$(ls ${TR_GLOB}*.jsonl.gz | paste -sd,)" \
            --only-deck dragapult_dusknoir --budget 4000 --per-game "$PER_GAME" \
            --margin-min 0.01 --playouts 16 --workers 12 --seed 43000 --out "$PAIRS" \
            || { say "local branch FAILED"; exit 1; }
    fi
fi
NP=$(zcat "$PAIRS" | wc -l)
say "pairs: $NP"
# 397 pairs came back from 240 games -- a quarter of what the field chain gets from 800.
# That is itself a datum: in a matchup lost 96% of the time, dpo_branch finds very few
# decisions where the alternative changes anything. Lowered so the ROUND can run; the pair
# count is reported with the result rather than used as a reason to stop.
[ "$NP" -ge 250 ] || { say "STOP: only $NP pairs"; exit 1; }

# ---------------------------------------------------------------- 3. probe + train
gpu_wait
say "probe"
python3 tools/instance/dpo_teacher.py --data "$PAIRS" --init-from "$PREV" \
    --card-first "$VOCAB" --out /root/out/probe_$TAG --probe --lr 5e-5 \
    > "$STATE/probe_$TAG.log" 2>&1
PACC=$(grep -a "^FINAL train loss" "$STATE/probe_$TAG.log" | sed -n "s/.*acc \([0-9.]*\)%.*/\1/p" | head -1)
say "probe train acc ${PACC:-?}%"
if ! awk -v a="${PACC:-0}" "BEGIN{exit !(a+0 >= 85)}"; then
    say "PROBE FAILED -- the 4B cannot even fit these pairs. That is itself the answer."
    tail -4 "$STATE/probe_$TAG.log"; exit 1
fi
say "train: init $PREV, reference pinned at $REF"
python3 tools/instance/dpo_teacher.py --data "$PAIRS" --init-from "$PREV" --ref-from "$REF" \
    --card-first "$VOCAB" --out "$OUT" --epochs 3 --beta 0.1 --lr 5e-5 --cdpo-calibrated \
    > "$STATE/train_$TAG.log" 2>&1 || { say "train FAILED"; tail -6 "$STATE/train_$TAG.log"; exit 1; }
grep -aE "\[ref\]|FINAL|saved" "$STATE/train_$TAG.log" | tail -4
[ -f "$OUT/domain_embeddings.pt" ] || { say "STOP: no checkpoint in $OUT"; exit 1; }

# ---------------------------------------------------------------- 4. gate
gpu_wait
say "gate: dusknoir cur vs new, $GATE_GAMES paired games each vs $OPP (reg)"
python3 -u tools/gate_protagonist.py --deck dragapult_dusknoir --opp "$OPP" \
    --games "$GATE_GAMES" --seed $((62000 + N * 100)) --baseline cur --opp-spec reg \
    --arm "cur=qwen:$PREV" --arm "new=qwen:$OUT" --mirror-so "$SO" \
    --out "$STATE/gate_$TAG.json" > "$STATE/gate_$TAG.log" 2>&1 \
    || { say "gate FAILED"; tail -10 "$STATE/gate_$TAG.log"; exit 1; }
grep -aE "vs |delta|arm |^cur|^new" "$STATE/gate_$TAG.log" | tail -8

python3 - "$STATE/gate_$TAG.json" "$STATE/baseA_$TAG.json" <<'PY'
import json, sys, os
g = json.load(open(sys.argv[1]))
cur, new = g["arms"]["cur"], g["arms"]["new"]
print("\n================ RESULT ================")
if os.path.exists(sys.argv[2]):
    a = json.load(open(sys.argv[2]))["arms"]["cur"]
    print("4B dusknoir vs engine_v2 ogerpon : %.1f%%  (n=%d)" % (a["win_rate"], a["games"]))
print("4B dusknoir vs 4B ogerpon        : %.1f%%  (n=%d)" % (cur["win_rate"], cur["games"]))
print("after one RL round               : %.1f%%   delta %+.2f +- %.2f (t %+.2f)"
      % (new["win_rate"], new["delta_vs_baseline"], new["se"],
         new["delta_vs_baseline"] / new["se"] if new["se"] else 0.0))
print("\nfor reference, instance1's cross-encoder vs engine_v2 ogerpon: 13.3%% (n=30)")
print("A 4B that cannot move this either is evidence the DECK loses this matchup, not the pilot.")
PY
say "DUSK_VS_OGER_DONE"
