#!/usr/bin/env bash
# One round of OPPONENT-adapter training for one deck.
#
# WHAT THIS IS FOR. The submission is dragapult_dusknoir piloted by the DeBERTa cross-encoder
# (mrl_r2). These Qwen adapters are its OPPOSITION: the five decks that make up 71% of the
# Kaggle top-100, each piloted by its own adapter, so dusknoir is trained and measured against
# decks that are PLAYED rather than merely held. They never ship -- a 4B does not fit the
# 197.66 MiB cap ([[submission-size-limit-lfm2-pivot]]).
#
# WHY BOTH SIDES COME FROM THE REGISTRY. The two seats need DIFFERENT prompt formats (dusknoir's
# DeBERTa renders DUSK_FMT, the Qwen renders PROMPT_FMT) and lm_mirror_log has ONE --fmt flag.
# models/adapters.json carries the format per deck and mirror_match's "reg" spec applies it per
# build, which is the only way these two models can share a game.
#
#   bash /root/deck_lora2.sh marnie_grimmsnarl 1
#
# Adds PER_GAME. dpo_branch samples at most --per-game branch points from each game,
# which is 11% of a 130-decision game and 1.5% of a slowking one: slowking runs 1,016
# decisions per game (measured over 150 games -- 7x every other deck) and yielded 1.13
# pairs per 1,000 decisions against 8-9 for the rest. Left at 15 it would collect for
# hours and come back under the 500-pair floor.
set -u
DECK=${1:?usage: deck_lora.sh <deck> <round>}
N=${2:?usage: deck_lora.sh <deck> <round>}
REPO=/root/ptcg/repo
VOCAB=$REPO/data/cardfirst_b_v39.json
REF=/root/out/i2_r7              # DPO beta anchor: the SFT, never re-anchored
STATE=/root/loop_deck
SO=$REPO/data/kaggle_engine_ext/libcg_mirror.so
GAMES=${GAMES:-150}              # per shard; 3 shards
BUDGET=${BUDGET:-12000}
PER_GAME=${PER_GAME:-15}         # branch points per game; slowking needs far more
PLAYOUTS=${PLAYOUTS:-24}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
mkdir -p "$STATE"; cd "$REPO"
say() { echo "[$DECK r$N $(date -u +%m-%d_%H:%M:%S)] $*"; }

TAG=${DECK}_r$N
TR_GLOB=/root/traces_$TAG.s
PAIRS=/root/pairs_$TAG.jsonl.gz
OUT=/root/out/lora_${DECK}_r$N

# The adapter this round starts from: last round's, or the fleet checkpoint for round 1. dpo_r7
# already pilots all eleven decks, so round 1 specialises an able generalist rather than
# starting from the SFT and re-learning what it already knows.
PREV=${PREV:-}
if [ -z "$PREV" ]; then
    if [ "$N" -gt 1 ]; then
        PREV=$(cat "$STATE/adopt_${DECK}_r$((N-1)).txt" 2>/dev/null || true)
        [ -n "$PREV" ] || { say "STOP: no adopt file for round $((N-1))"; exit 1; }
        PREV=/root/out/$PREV
    else
        PREV=/root/out/dpo_r7
    fi
fi
[ -d "$PREV" ] || { say "STOP: no checkpoint at $PREV"; exit 1; }
[ -d /root/out/mrl_r2 ] || { say "STOP: dusknoir's champion mrl_r2 is not on this machine"; exit 1; }
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

# The registry is what makes both sides loadable at once, and it is also the round's state: the
# collection reads THIS deck's current adapter from it, so it must be written before collecting.
python3 tools/adapters.py set "$DECK" --target "qwen:$(basename $PREV)" --fmt prompt \
    --note "opponent adapter, round $N init" || exit 1
python3 tools/adapters.py check || true

# ---------------------------------------------------------------- 1. collect
# Resume on SIZE, not existence: a killed shard leaves a 0-byte .gz behind, and the old
# existence test then skipped collection and shipped empty traces to the brancher (this is
# exactly how slowking r1 burned 2h waiting for a branch that could never be built).
if ! { [ -s ${TR_GLOB}0.jsonl.gz ] && [ -s ${TR_GLOB}1.jsonl.gz ] && [ -s ${TR_GLOB}2.jsonl.gz ]; }; then
    rm -f ${TR_GLOB}*.jsonl.gz
    gpu_wait
    say "collect: $DECK (reg) vs dragapult_dusknoir (mrl_r2), 3 x $GAMES games"
    j=0
    for SH in 0 1 2; do
        PYTHONPATH=cg-lib nohup python3 tools/lm_mirror_log.py \
            --model reg --deck-model reg \
            --protagonist dragapult_dusknoir --decks "$DECK" --games "$GAMES" \
            --seed $((300000 + N * 10000 + SH * 1000)) \
            --out /root/lmlog_$TAG.s$SH.jsonl.gz --trace-out ${TR_GLOB}$SH.jsonl.gz \
            --mirror-so "$SO" > /root/collect_$TAG.s$SH.log 2>&1 &
        j=$((j+1))
    done
    say "launched $j collection shards"; wait
fi
ls ${TR_GLOB}*.jsonl.gz >/dev/null 2>&1 || { say "STOP: no traces"; exit 1; }

# ---------------------------------------------------------------- 2. branch via instance1
# Resume on RECORD COUNT, not file size. A gzip holding ZERO records is ~53 bytes, so -s is
# true for it: pass 3 skipped the branch step entirely for four decks because pass 2 had
# shipped them empty pairs files, then died at the count check below with 'pairs: 0' and no
# hint of why. Exactly the trap the trace guard above was written for, one file downstream.
if [ "$(zcat "$PAIRS" 2>/dev/null | head -1 | wc -l)" -eq 0 ]; then
    rm -f "$PAIRS"
    echo "$TAG|$DECK|$BUDGET|$PLAYOUTS|$PER_GAME" > /root/branch_request2
    # Empty traces cannot produce pairs; fail here with a readable message rather than
    # letting the brancher fail 2h later on the far side of the link.
    for SH in 0 1 2; do
        [ -s ${TR_GLOB}$SH.jsonl.gz ] || { say "STOP: trace shard $SH is empty -- collection failed, see /root/collect_$TAG.s$SH.log"; exit 1; }
    done
    say "branch requested from instance1 (tag $TAG, budget $BUDGET, playouts $PLAYOUTS, per-game $PER_GAME)"
    for _ in $(seq 1 120); do
        [ -s "$PAIRS" ] && break
        sleep 60
    done
    if [ ! -s "$PAIRS" ]; then
        # LOUD, never silent: 13.44 effective cores is a fifth of instance1's, so the local
        # fallback buys a SMALLER and NOISIER pair set. Any verdict read off it must know.
        say "FALLBACK: instance1 did not deliver in 2h -- building LOCALLY at budget 4000,"
        say "FALLBACK: playouts 16. This round's pairs are SMALLER AND NOISIER than the others."
        rm -f /root/branch_request2
        RL_PRIZE_GAMMA=0.25 CUDA_VISIBLE_DEVICES= PYTHONPATH=cg-lib nice -n 5 \
            python3 tools/dpo_branch.py --traces "$(ls ${TR_GLOB}*.jsonl.gz | paste -sd,)" \
            --only-deck "$DECK" --budget 4000 --per-game "$PER_GAME" --margin-min 0.01 \
            --playouts 16 --workers 12 --seed 41000 --out "$PAIRS" \
            || { say "local branch FAILED too"; exit 1; }
    fi
fi
NP=$(zcat "$PAIRS" | wc -l)
say "pairs: $NP"
[ "$NP" -ge 500 ] || { say "STOP: only $NP pairs"; exit 1; }

# ---------------------------------------------------------------- 3. probe + train
gpu_wait
say "probe"
python3 tools/instance/dpo_teacher.py --data "$PAIRS" --init-from "$PREV" \
    --card-first "$VOCAB" --out /root/out/probe_$TAG --probe --lr 5e-5 \
    > "$STATE/probe_$TAG.log" 2>&1
# Gate on ACCURACY, not on the 0.15 loss threshold dpo_teacher prints its verdict from.
# slowking r1 overfit 2464 pairs to 94.1% train accuracy at loss 0.1644 and was killed for
# missing 0.15 by 0.014 -- the probe exists to catch a trainer that CANNOT learn, and 94%
# is not that. Loss floors move with pair difficulty per deck; accuracy does not.
PACC=$(grep -a "^FINAL train loss" "$STATE/probe_$TAG.log" | sed -n "s/.*acc \([0-9.]*\)%.*/\1/p" | head -1)
if ! awk -v a="${PACC:-0}" "BEGIN{exit !(a+0 >= 85)}"; then
    say "PROBE FAILED (train acc ${PACC:-?}% < 85)"; tail -4 "$STATE/probe_$TAG.log"; exit 1
fi
say "probe OK (train acc ${PACC}%)"
say "train: init $PREV, reference pinned at $REF"
python3 tools/instance/dpo_teacher.py --data "$PAIRS" --init-from "$PREV" --ref-from "$REF" \
    --card-first "$VOCAB" --out "$OUT" --epochs 3 --beta 0.1 --lr 5e-5 --cdpo-calibrated \
    > "$STATE/train_$TAG.log" 2>&1 || { say "train FAILED"; tail -6 "$STATE/train_$TAG.log"; exit 1; }
grep -aE "\[ref\]|FINAL|saved" "$STATE/train_$TAG.log" | tail -4
[ -f "$OUT/domain_embeddings.pt" ] || { say "STOP: no checkpoint in $OUT"; exit 1; }

# ---------------------------------------------------------------- 4. gate
# The question is "does this deck now beat dusknoir MORE OFTEN", so both arms pilot $DECK and
# the opponent is dusknoir played by its champion (--opp-spec reg -> hf:mrl_r2). Paired on
# (seed, seat), so the two arms meet the identical shuffles.
gpu_wait
# 200 games/arm, not 400: a Qwen decision is ~130 ms and a game is ~130 of them, so the arm
# side alone costs ~17 s/game -- 400 would put a 3.8 h gate on top of a 2.4 h round. These are
# SPARRING PARTNERS, and a 2pt-weaker partner is a minor loss, so the check is sized to catch
# a collapse rather than to resolve small edges.
GATE_GAMES=${GATE_GAMES:-200}
say "gate: $DECK new vs cur, $GATE_GAMES paired games each vs dusknoir/mrl_r2"
python3 -u tools/gate_protagonist.py --deck "$DECK" --opp dragapult_dusknoir \
    --games "$GATE_GAMES" --seed $((51000 + N * 100)) --baseline cur --opp-spec reg \
    --arm "cur=qwen:$PREV" --arm "new=qwen:$OUT" --mirror-so "$SO" \
    --out "$STATE/gate_$TAG.json" > "$STATE/gate_$TAG.log" 2>&1 \
    || { say "gate FAILED"; tail -10 "$STATE/gate_$TAG.log"; exit 1; }
grep -aE "vs |delta|arm " "$STATE/gate_$TAG.log" | tail -6

VERDICT=$(python3 - "$STATE/gate_$TAG.json" <<'PY'
import json, sys
j = json.load(open(sys.argv[1]))
new = (j.get("arms") or {}).get("new", {})
d, se = new.get("delta_vs_baseline", 0.0), new.get("se", 0.0)
t = d / se if se else 0.0
# REJECT on the POINT ESTIMATE, not on t. The mirror chain adopted a round that measured
# -5.00pt because the rule demanded d<=-2 AND t<=-2, and at n=320 (SE 3.6pt) t=-2 needs
# -7.2pt -- every drop between -2 and -7pt passed. At 200 paired games SE is ~3.9pt, so the
# threshold is set at -3: it still rejects a collapse, and a false stop is cheap (the previous
# adapter is kept and this deck simply stops here).
print("ADOPT" if d > -3.0 else "REJECT")
print("delta %+.2f +- %.2f (t %+.2f)" % (d, se, t), file=sys.stderr)
PY
)
say "round $N verdict: $VERDICT"
if [ "$VERDICT" = "ADOPT" ]; then
    echo "lora_${DECK}_r$N" > "$STATE/adopt_${DECK}_r$N.txt"
    python3 tools/adapters.py set "$DECK" --target "qwen:lora_${DECK}_r$N" --fmt prompt \
        --note "opponent adapter, round $N adopted"
    W=$(python3 -c "
import json;j=json.load(open('$STATE/gate_$TAG.json'));print('%.4f'%j['arms']['new']['win_rate'])")
    python3 tools/adapters.py gate "$DECK" --win "$W" --games "$GATE_GAMES" \
        --opp dragapult_dusknoir --vs mrl_r2 --date "$(date -u +%Y-%m-%d)"
else
    say "round $N REJECTED -- registry left pointing at $PREV"
    python3 tools/adapters.py set "$DECK" --target "qwen:$(basename $PREV)" --fmt prompt \
        --note "opponent adapter, round $N rejected; kept round $((N-1))"
fi
say "DECK_LORA_DONE $DECK r$N $VERDICT"
