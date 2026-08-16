#!/usr/bin/env bash
# Mirror RL v3: the five arithmetic rules pilot their menus, the model pilots the rest.
#
# v2 stopped itself at 12:12 after three rounds without a gain (rounds 6-8, champion mrl2_r5b
# throughout) -- at lr 2e-6 the training is finally reproducible and what it reproduces is
# "no further gain from this data". Two things change at once here, deliberately:
#
#   * DEFERRAL IS IN THE LOOP, not bolted on after. Collection, both challengers and the gate
#     all run through planfilter:R5, so the model trains and is measured on exactly the state
#     distribution it will see deployed. R5 = lethal_now, spread_aim, clops_hold, energy_line,
#     energy_focus -- the five whose answer is a subtraction, measured 0.02x-1.76x with the
#     model taking 1.5%-43% of them. The one-shot deferral gates kept landing at +1..2pt +-2.2;
#     if the effect is real it compounds through the LOOP, and that is the honest test of it.
#   * The reward question stays an A/B: challenger a trains on Q+prizes alone (beta 0),
#     challenger b blends in conformance to the ELEVEN rules the model still owns (beta 0.3).
#     The five deferred rules are excluded from rww/rwl -- the model never makes those
#     decisions, so gradient toward them is spent on nothing.
#
#   step 0  gate: bare champion vs deferral-wrapped champion, 600 paired mirror games.
#           Chooses the wrapper for the whole chain; a deferral more than 2pt WORSE is refused.
#   rounds  collect(planfilter) -> branch(rule weights, minus R5) -> a/b train -> 3-arm gate
set -u
REPO=/root/ptcg/repo; cd "$REPO"
export PYTHONPATH=cg-lib HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export RL_PRIZE_GAMMA=0.25
export DUSK_NEW_RULES=1 DUSK_CLOPS_HOLD=1     # the new rules exist only under these flags
SO=$REPO/data/kaggle_engine_ext/libcg_mirror.so
STATE=/root/loop_dusk/mrl3
DECK=dragapult_dusknoir
R5=lethal_now,spread_aim,clops_hold,energy_line,energy_focus
CUR=${CUR:-/root/out/mrl2_r5b}
FROM=${FROM:-1}
ROUNDS=${ROUNDS:-6}
GATE_GAMES=${GATE_GAMES:-600}
mkdir -p "$STATE"
say() { echo "[mrl3 $(date -u +%m-%d_%H:%M:%S)] $*"; }

gpu_wait() {
    local u
    for _ in $(seq 1 60); do
        u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
        [ "$u" -le 2000 ] && return 0
        sleep 30
    done
    say "STOP: GPU held ${u} MiB for 30 min"; exit 1
}

# ---------------------------------------------------------------- 0. wrapper gate
if [ ! -s "$STATE/wrap.txt" ]; then
    gpu_wait
    say "wrapper gate: bare vs planfilter:R5, $GATE_GAMES paired mirror games"
    python3 -u tools/gate_protagonist.py --deck "$DECK" --opp "$DECK" --games "$GATE_GAMES" \
        --seed 95000 --baseline bare \
        --arm "bare=hf:$CUR@dusk" \
        --arm "def=planfilter:$R5:hf:$CUR@dusk" \
        --mirror-so "$SO" --out "$STATE/gate_wrap.json" > "$STATE/gate_wrap.log" 2>&1 \
        || { say "wrapper gate FAILED"; tail -8 "$STATE/gate_wrap.log"; exit 1; }
    grep -aE "vs |delta" "$STATE/gate_wrap.log" | tail -4
    python3 - "$STATE/gate_wrap.json" > "$STATE/wrap.txt" <<'PY'
import json, sys
d = (json.load(open(sys.argv[1]))["arms"].get("def") or {}).get("delta_vs_baseline")
# The wrapper must merely not hurt: its value compounds through the loop, so it is kept unless
# it is measurably worse on its own. -2pt at SE ~2.2 is the same bar the champion rule uses.
print("def" if (d is not None and d > -2.0) else "bare")
print("deferral delta %+.2f" % (d if d is not None else float("nan")), file=sys.stderr)
PY
fi
WRAP=$(head -1 "$STATE/wrap.txt")
PFX=""
RULE_EXCL="$R5"
if [ "$WRAP" = "def" ]; then
    PFX="planfilter:$R5:"
else
    # BARE MODE PUTS THE FIVE RULES BACK IN THE REWARD. The exclusion exists because a deferred
    # decision is never the model's, so gradient toward it is wasted -- but if the wrapper gate
    # rejected deferral, the model IS making those decisions again, and excluding them would
    # silently delete exactly the five signals (0.02x-0.51x execution) most worth teaching.
    RULE_EXCL=""
fi
say "wrapper: $WRAP (prefix '$PFX', reward excludes '${RULE_EXCL:-none}')"

# ---------------------------------------------------------------- rounds
MISSES=0
for R in $(seq "$FROM" "$ROUNDS"); do
    say "================ v3 round $R (champion $CUR) ================"
    TR=/root/mrl3_tr$R.jsonl.gz
    PAIRS=/root/mrl3_pairs$R.jsonl.gz

    if [ ! -s "$TR" ]; then
        gpu_wait
        say "collect: 400 mirror games through '$WRAP'"
        python3 tools/lm_mirror_log.py --model "${PFX}hf:$CUR" --fmt dusk \
            --protagonist "$DECK" --decks "$DECK" --games 400 --seed $((500000 + R * 1000)) \
            --out /root/mrl3_log$R.jsonl.gz --trace-out "$TR" --mirror-so "$SO" \
            > "$STATE/collect$R.log" 2>&1 \
            || { say "collect FAILED"; tail -8 "$STATE/collect$R.log"; exit 1; }
    fi

    if [ ! -s "$PAIRS" ]; then
        say "branch: rule weights on, R5 excluded from them"
        CUDA_VISIBLE_DEVICES= nice -n 5 python3 tools/dpo_branch.py \
            --traces "$TR" --fmt dusk --rule-weights --rule-exclude "$RULE_EXCL" \
            --budget 6000 --per-game 15 --margin-min 0.01 --playouts 24 --workers 24 \
            --out "$PAIRS" > "$STATE/branch$R.log" 2>&1 \
            || { say "branch FAILED"; tail -8 "$STATE/branch$R.log"; exit 1; }
        grep -aE "^wrote|selected" "$STATE/branch$R.log" | tail -2
        python3 - "$PAIRS" <<'PY'
import gzip, json, sys
n = nz = 0
for line in gzip.open(sys.argv[1], "rt"):
    d = json.loads(line); n += 1
    if (d.get("rww") or 0) > 0 or (d.get("rwl") or 0) > 0:
        nz += 1
print("[rules] %d/%d pairs carry a rule weight (%.1f%%)%s"
      % (nz, n, 100.0 * nz / max(n, 1),
         "" if nz else "  <-- ZERO: arm b is label smoothing, NOT rule conformance"))
PY
    fi

    for V in a b; do
        BETA=0.0; [ "$V" = "b" ] && BETA=0.3
        ROWS=$STATE/rows_r$R$V.jsonl.gz
        python3 /root/mrl_convert.py --pairs "$PAIRS" --out "$ROWS" \
            --beta "$BETA" --temp 0.5 | tee -a "$STATE/convert$R.log"
        NR=$(zcat "$ROWS" | wc -l)
        [ "$NR" -ge 500 ] || { say "STOP: only $NR rows"; exit 1; }
        gpu_wait
        say "train $V: beta $BETA, $NR rows, lr 2e-6 ep 0.5 l2sp 1e-2"
        python3 tools/dusk_plan_train.py --data "$ROWS" --model "$CUR" \
            --out /root/out/mrl3_r$R$V --lr 2e-6 --epochs 0.5 --accum 4 --l2sp 1e-2 \
            > "$STATE/train$R$V.log" 2>&1 \
            || { say "train $V FAILED"; tail -6 "$STATE/train$R$V.log"; exit 1; }
        grep -aE "FINAL|\[eval\]" "$STATE/train$R$V.log" | tail -2
        [ -f "/root/out/mrl3_r$R$V/model.safetensors" ] || { say "STOP: no checkpoint $V"; exit 1; }
    done

    gpu_wait
    say "gate: champion vs a vs b, all through '$WRAP', $GATE_GAMES paired games"
    python3 -u tools/gate_protagonist.py --deck "$DECK" --opp "$DECK" --games "$GATE_GAMES" \
        --seed $((71000 + R * 100)) --baseline cur \
        --arm "cur=${PFX}hf:$CUR@dusk" \
        --arm "a=${PFX}hf:/root/out/mrl3_r${R}a@dusk" \
        --arm "b=${PFX}hf:/root/out/mrl3_r${R}b@dusk" \
        --mirror-so "$SO" --out "$STATE/gate_r$R.json" > "$STATE/gate_r$R.log" 2>&1 \
        || { say "gate FAILED"; tail -10 "$STATE/gate_r$R.log"; exit 1; }
    grep -aE "vs |delta|^arm|^a |^b |^cur " "$STATE/gate_r$R.log" | tail -8

    WIN=$(python3 - "$STATE/gate_r$R.json" <<'PY'
import json, sys
arms = json.load(open(sys.argv[1])).get("arms", {})
best, bd = None, None
for k in ("a", "b"):
    d = (arms.get(k) or {}).get("delta_vs_baseline")
    if d is not None and (bd is None or d > bd):
        best, bd = k, d
print(best if (bd is not None and bd > 1.0) else "none")
print("a %+.2f | b %+.2f" % ((arms.get("a") or {}).get("delta_vs_baseline", float("nan")),
                             (arms.get("b") or {}).get("delta_vs_baseline", float("nan"))),
      file=sys.stderr)
PY
)
    say "round $R winner: $WIN"
    if [ "$WIN" = "none" ]; then
        MISSES=$((MISSES+1))
        say "round $R: no challenger cleared +1.0pt -- champion stays $CUR ($MISSES in a row)"
        [ "$MISSES" -ge 3 ] && { say "three misses -- stopping for a human read"; break; }
    else
        MISSES=0
        CUR=/root/out/mrl3_r$R$WIN
        echo "$CUR" > "$STATE/current.txt"
        say "new champion: $CUR"
        python3 tools/adapters.py set "$DECK" --target "hf:$(basename $CUR)" --fmt dusk \
            --wrap "planfilter:$R5" \
            --note "mirror v3 champion, round $R arm $WIN" || true
    fi
done
say "MIRROR_CHAIN3_DONE (champion: $CUR, wrapper: $WRAP)"
