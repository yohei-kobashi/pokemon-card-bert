#!/usr/bin/env bash
# Mirror RL, CHAMPION vs CHALLENGERS.
#
# WHY THE SHAPE CHANGED. The v1 chain adopted every round that was not catastrophically worse,
# so the champion did a random walk: +9.06, +3.12, -5.00, and the 600-game head-to-head then
# showed round 3 sitting 6.17pt (t +2.36) BELOW the round it had replaced. Two changes follow:
#
#   * The champion only ever moves UP. A challenger must measure better than the champion by
#     more than ADOPT_MIN to replace it; otherwise the champion stands and the round is spent.
#     Never adopting a regression is worth more than adopting every small real gain, because a
#     regression is carried forward into every later round's data.
#   * TWO challengers per round, trained from the same champion on the same pairs, gated
#     together. If the round-to-round swing is optimisation variance rather than signal (which
#     /root/loop_dusk/r3diag was built to decide), sampling twice and keeping the better one
#     turns that variance from a liability into a source of gain.
#
# The two challengers differ in ONE thing: whether dusk_plan rule conformance is in the label.
#   A  beta 0            labels from the playout Q alone
#   B  beta 0.3          Q blended with the rule weights the user asked for
# That comparison was impossible until today: --rule-weights emitted rww=rwl=0 on every pair
# because the worker pool is a SPAWN context and the flag lived in a module global assigned in
# main(). With that fixed, B is the first honest test of the rule-conformance term, and it has
# to win its place rather than be assumed.
set -u
REPO=/root/ptcg/repo; cd "$REPO"
export PYTHONPATH=cg-lib HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export RL_PRIZE_GAMMA=0.25          # terminal prize-margin shaping (env-local; branchd unaffected)
SO=$REPO/data/kaggle_engine_ext/libcg_mirror.so
STATE=/root/loop_dusk/mrl2
DECK=dragapult_dusknoir
CUR=${CUR:?set CUR to the champion checkpoint}
TEMP=${TEMP:-0.5}                   # label sharpness, chosen by the r3diag gate
FROM=${FROM:-4}
ROUNDS=${ROUNDS:-8}
GAMES=${GAMES:-400}                 # mirror games collected per round
GATE_GAMES=${GATE_GAMES:-600}       # SE ~2.6pt
ADOPT_MIN=${ADOPT_MIN:-1.0}         # a challenger must be AHEAD, not merely not-behind
mkdir -p "$STATE"
say() { echo "[mrl2 $(date -u +%m-%d_%H:%M:%S)] $*"; }

gpu_wait() {
    local u
    for _ in $(seq 1 60); do
        u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
        [ "$u" -le 2000 ] && return 0
        sleep 30
    done
    say "STOP: GPU held ${u} MiB for 30 min"; exit 1
}

say "champion $CUR | temp $TEMP | rounds $FROM..$ROUNDS | adopt if > +${ADOPT_MIN}pt"
MISSES=0
for R in $(seq "$FROM" "$ROUNDS"); do
    say "================ round $R (champion $CUR) ================"
    TR=/root/mrl2_tr$R.jsonl.gz
    PAIRS=/root/mrl2_pairs$R.jsonl.gz

    if [ ! -s "$TR" ]; then
        gpu_wait
        say "collect: $GAMES mirror games"
        python3 tools/lm_mirror_log.py --model "hf:$CUR" --fmt dusk \
            --protagonist "$DECK" --decks "$DECK" --games "$GAMES" \
            --seed $((400000 + R * 1000)) \
            --out /root/mrl2_log$R.jsonl.gz --trace-out "$TR" --mirror-so "$SO" \
            > "$STATE/collect$R.log" 2>&1 \
            || { say "collect FAILED"; tail -8 "$STATE/collect$R.log"; exit 1; }
    fi

    if [ ! -s "$PAIRS" ]; then
        say "branch: prize-shaped playouts, rule weights ON (they reach the workers now)"
        CUDA_VISIBLE_DEVICES= nice -n 5 python3 tools/dpo_branch.py \
            --traces "$TR" --fmt dusk --rule-weights \
            --budget 6000 --per-game 15 --margin-min 0.01 --playouts 24 --workers 24 \
            --out "$PAIRS" > "$STATE/branch$R.log" 2>&1 \
            || { say "branch FAILED"; tail -8 "$STATE/branch$R.log"; exit 1; }
        grep -aE "^wrote|selected" "$STATE/branch$R.log" | tail -2
        # The bug this loop was built around: if the rule weights are all zero again, the beta
        # arm is not testing rule conformance -- it is smoothing labels 30% toward uniform, and
        # saying so is the whole point of having found it once.
        python3 - "$PAIRS" <<'PY'
import gzip, json, sys
n = nz = 0
for line in gzip.open(sys.argv[1], "rt"):
    d = json.loads(line); n += 1
    if (d.get("rww") or 0) > 0 or (d.get("rwl") or 0) > 0:
        nz += 1
print("[rules] %d/%d pairs carry a rule weight (%.1f%%)%s"
      % (nz, n, 100.0 * nz / max(n, 1),
         "" if nz else "  <-- ZERO: the beta arm is label smoothing, NOT rule conformance"))
PY
    fi

    NEWA=/root/out/mrl2_r${R}a
    NEWB=/root/out/mrl2_r${R}b
    for V in a b; do
        BETA=0.0; [ "$V" = "b" ] && BETA=0.3
        ROWS=$STATE/rows_r$R$V.jsonl.gz
        python3 /root/mrl_convert.py --pairs "$PAIRS" --out "$ROWS" \
            --beta "$BETA" --temp "$TEMP" | tee -a "$STATE/convert$R.log"
        NR=$(zcat "$ROWS" | wc -l)
        [ "$NR" -ge 500 ] || { say "STOP: only $NR rows"; exit 1; }
        EPOCHS=$(python3 -c "print(round(min(2.0, 8000.0/$NR), 2))")
        # The sweep pins these when it finds a setting whose two row orders agree;
        # unset, the round keeps the original recipe.
        [ -n "${EPOCHS_FIX:-}" ] && EPOCHS=$EPOCHS_FIX
        gpu_wait
        say "train $V: beta $BETA temp $TEMP, $NR rows, epochs $EPOCHS, lr ${LR:-1e-5}, l2sp ${L2SP:-1e-3}"
        python3 tools/dusk_plan_train.py --data "$ROWS" --model "$CUR" \
            --out /root/out/mrl2_r$R$V --lr "${LR:-1e-5}" --epochs "$EPOCHS" --accum 4 \
            --l2sp "${L2SP:-1e-3}" \
            > "$STATE/train$R$V.log" 2>&1 \
            || { say "train $V FAILED"; tail -6 "$STATE/train$R$V.log"; exit 1; }
        grep -aE "FINAL|\[eval\]" "$STATE/train$R$V.log" | tail -2
        [ -f "/root/out/mrl2_r$R$V/model.safetensors" ] || { say "STOP: no checkpoint $V"; exit 1; }
    done

    gpu_wait
    say "gate: champion vs both challengers, $GATE_GAMES paired mirror games each"
    python3 -u tools/gate_protagonist.py --deck "$DECK" --opp "$DECK" --games "$GATE_GAMES" \
        --seed $((61000 + R * 100)) --baseline cur \
        --arm "cur=hf:$CUR@dusk" --arm "a=hf:$NEWA@dusk" --arm "b=hf:$NEWB@dusk" \
        --mirror-so "$SO" --out "$STATE/gate_r$R.json" > "$STATE/gate_r$R.log" 2>&1 \
        || { say "gate FAILED"; tail -10 "$STATE/gate_r$R.log"; exit 1; }
    grep -aE "vs |delta|^arm|^a |^b |^cur " "$STATE/gate_r$R.log" | tail -8

    WIN=$(python3 - "$STATE/gate_r$R.json" "$ADOPT_MIN" <<'PY'
import json, sys
j = json.load(open(sys.argv[1]))
lo = float(sys.argv[2])
arms = j.get("arms", {})
best, bd = None, None
for k in ("a", "b"):
    d = (arms.get(k) or {}).get("delta_vs_baseline")
    if d is None:
        continue
    if bd is None or d > bd:
        best, bd = k, d
# The champion moves only on a measured gain. Picking the max of two challengers biases the
# estimate up by roughly 0.56 SE (~1.5pt here), which is exactly why the bar is a gain and not
# merely "not a loss" -- the v1 rule adopted anything above -2pt and walked downhill.
print(best if (bd is not None and bd > lo) else "none")
print("a %+.2f | b %+.2f | keep %.1f" % (
    (arms.get("a") or {}).get("delta_vs_baseline", float("nan")),
    (arms.get("b") or {}).get("delta_vs_baseline", float("nan")), lo), file=sys.stderr)
PY
)
    say "round $R winner: $WIN"
    if [ "$WIN" = "none" ]; then
        MISSES=$((MISSES+1))
        say "round $R: neither challenger cleared +${ADOPT_MIN}pt -- champion stays $CUR ($MISSES in a row)"
        [ "$MISSES" -ge 3 ] && { say "three rounds without a gain -- stopping for a human read"; break; }
    else
        MISSES=0
        CUR=/root/out/mrl2_r$R$WIN
        echo "$CUR" > "$STATE/current.txt"
        say "new champion: $CUR"
        python3 tools/adapters.py set "$DECK" --target "hf:$(basename $CUR)" --fmt dusk \
            --note "mirror champion, round $R arm $WIN" || true
        W=$(python3 -c "
import json;j=json.load(open('$STATE/gate_r$R.json'));print('%.2f'%j['arms']['$WIN']['win_rate'])")
        python3 tools/adapters.py gate "$DECK" --win "$W" --games "$GATE_GAMES" \
            --opp "$DECK" --vs engine_v2 --date "$(date -u +%Y-%m-%d)" || true
    fi
done
say "MIRROR_CHAIN2_DONE (champion: $CUR)"
