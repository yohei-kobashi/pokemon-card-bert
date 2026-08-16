#!/usr/bin/env bash
# Overnight mirror-RL chain on instance1 (user 2026-08-10):
#
#   "dusknoir mirror ONLY; reward = win/loss + prizes + rule conformance; evaluate WITH the
#    rule-based deferral, against engine_v2, mirror only."
#
#   0. wait for the rule-deferral gate, pick the wrapper (strict/filter/none) from its result
#   1. baseline: s1(+wrapper) vs engine_v2 in the dusknoir mirror
#   2. rounds: collect (wrapped pilot, GPU) -> branch (prize-shaped playouts + rule weights,
#      CPU) -> convert (beta-blend) -> train (fp32, lr 1e-5, anchored, step-capped) -> gate
#      (mirror, paired) -> adopt/stop
#
# Steps are capped and gating is games-only: the plan-conformance collapse (18.9% -> 5.2% at
# 93.6% conformance) is the standing reason neither conformance nor loss ever decides adoption.
set -u
REPO=/root/ptcg/repo; cd "$REPO"
export PYTHONPATH=cg-lib HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export RL_PRIZE_GAMMA=0.25          # terminal prize-margin shaping; branchd is unaffected (env-local)
SO=$REPO/data/kaggle_engine_ext/libcg_mirror.so
STATE=/root/loop_dusk/mrl
DECK=dragapult_dusknoir
DEFER="spread_aim,energy_line,energy_focus,recon"
ROUNDS=${ROUNDS:-4}
FROM=${FROM:-1}                     # first round number to run (resume)
GATE_GAMES=${GATE_GAMES:-600}       # SE ~2.6pt; a DeBERTa-vs-engine game costs ~1.8 s
mkdir -p "$STATE"
say() { echo "[mrl $(date -u +%m-%d_%H:%M:%S)] $*"; }

gpu_wait() {
    local u
    for _ in $(seq 1 60); do
        u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
        [ "$u" -le 2000 ] && return 0
        sleep 30
    done
    say "STOP: GPU held ${u} MiB for 30 min"; exit 1
}

# ---------------------------------------------------------------- 0. wrapper choice
if [ ! -s "$STATE/wrapper.txt" ]; then
    say "waiting for the rule-deferral gate"
    while ! grep -q "GATE_RULES_DONE" /root/after_merge.log 2>/dev/null; do sleep 120; done
fi
python3 - "$DEFER" > "$STATE/wrapper.txt" <<'PY'
import glob, json, sys
rules = sys.argv[1]
cells = {}
for p in sorted(glob.glob("/root/loop_dusk/gate_rules/*.json")):
    try:
        cells.update(json.load(open(p)).get("cells", {}))
    except Exception:
        pass
def tot(arm):
    v = []
    for k, c in cells.items():
        if k.split("|")[0] == arm:
            v.extend(c.get("raw") or [])
    return v
base = tot("s1")
def delta(arm):
    v = tot(arm)
    if not v or len(v) != len(base):
        return None
    return 100.0 * sum(x - y for x, y in zip(v, base)) / len(v)
ds, df = delta("strict"), delta("filter")
best, name = 0.0, "none"
for d, n in ((ds, "planrule"), (df, "planfilter")):
    if d is not None and d > best:
        best, name = d, n
# The wrapper must EARN its place: at 480 paired games the SE is ~2pt, so a sub-1pt win is
# noise and 'none' (the plain model) is the honest default.
if best < 1.0:
    name = "none"
print(name)
print("strict %+0.2f | filter %+0.2f" % (ds if ds is not None else float("nan"),
                                         df if df is not None else float("nan")), file=sys.stderr)
PY
WRAP=$(head -1 "$STATE/wrapper.txt")
case "$WRAP" in
    planrule|planfilter) PFX="$WRAP:$DEFER:";;
    *)                   PFX="";;
esac
say "wrapper choice: ${WRAP} (spec prefix '${PFX}')"

CUR=${CUR:-/root/out/dusk_s1}

# ---------------------------------------------------------------- 1. baseline
if [ ! -s "$STATE/gate_base.json" ]; then
    gpu_wait
    say "baseline: s1(+wrapper) vs engine_v2, dusknoir mirror, 320 games"
    python3 -u tools/gate_protagonist.py --deck "$DECK" --opp "$DECK" --games 320 \
        --seed 21000 --baseline cur \
        --arm "cur=${PFX}hf:$CUR@dusk" \
        --out "$STATE/gate_base.json" > "$STATE/gate_base.log" 2>&1 \
        || { say "baseline gate FAILED"; tail -10 "$STATE/gate_base.log"; exit 1; }
    grep -a "vs " "$STATE/gate_base.log" | tail -2
fi

# ---------------------------------------------------------------- 2. rounds
MISSES=0
for R in $(seq "$FROM" "$ROUNDS"); do
    say "================ mirror round $R (from $CUR) ================"
    TR=/root/mrl_tr$R.jsonl.gz
    PAIRS=/root/mrl_pairs$R.jsonl.gz
    ROWS=/root/mrl_rows$R.jsonl.gz
    NEW=/root/out/mrl_r$R

    if [ ! -s "$TR" ]; then
        gpu_wait
        say "collect: 400 mirror games, wrapped pilot"
        python3 tools/lm_mirror_log.py --model "${PFX}hf:$CUR" --fmt dusk \
            --protagonist "$DECK" --decks "$DECK" --games 400 --seed $((200000 + R * 1000)) \
            --out /root/mrl_log$R.jsonl.gz --trace-out "$TR" --mirror-so "$SO" \
            > "$STATE/collect$R.log" 2>&1 \
            || { say "collect FAILED"; tail -8 "$STATE/collect$R.log"; exit 1; }
    fi

    if [ ! -s "$PAIRS" ]; then
        say "branch: prize-shaped playouts (gamma $RL_PRIZE_GAMMA), rule weights on"
        CUDA_VISIBLE_DEVICES= nice -n 5 python3 tools/dpo_branch.py \
            --traces "$TR" --fmt dusk --rule-weights \
            --budget 6000 --per-game 15 --margin-min 0.01 --playouts 24 --workers 24 \
            --out "$PAIRS" > "$STATE/branch$R.log" 2>&1 \
            || { say "branch FAILED"; tail -8 "$STATE/branch$R.log"; exit 1; }
        grep -aE "^wrote|pair " "$STATE/branch$R.log" | tail -3
    fi

    python3 /root/mrl_convert.py --pairs "$PAIRS" --out "$ROWS" --beta 0.3 --temp 0.5 \
        | tee -a "$STATE/convert$R.log"
    NR=$(zcat "$ROWS" | wc -l)
    [ "$NR" -ge 500 ] || { say "STOP: only $NR rows"; exit 1; }
    # Step cap: the plan collapse ran 72k steps; this loop never exceeds 8k.
    EPOCHS=$(python3 -c "print(round(min(2.0, 8000.0/$NR), 2))")

    gpu_wait
    say "train: $NR rows, epochs $EPOCHS (step cap 8k), lr 1e-5, anchored"
    python3 tools/dusk_plan_train.py --data "$ROWS" --model "$CUR" --out "$NEW" \
        --lr 1e-5 --epochs "$EPOCHS" --accum 4 --l2sp 1e-3 > "$STATE/train$R.log" 2>&1 \
        || { say "train FAILED"; tail -8 "$STATE/train$R.log"; exit 1; }
    grep -aE "\[data\]|\[eval\]|FINAL" "$STATE/train$R.log" | tail -3
    [ -f "$NEW/model.safetensors" ] || { say "STOP: no checkpoint"; exit 1; }

    gpu_wait
    say "gate: cur vs new, mirror, $GATE_GAMES paired games each vs engine_v2"
    python3 -u tools/gate_protagonist.py --deck "$DECK" --opp "$DECK" --games "$GATE_GAMES" \
        --seed $((21000 + R * 100)) --baseline cur \
        --arm "cur=${PFX}hf:$CUR@dusk" \
        --arm "new=${PFX}hf:$NEW@dusk" \
        --out "$STATE/gate_r$R.json" > "$STATE/gate_r$R.log" 2>&1 \
        || { say "gate FAILED"; tail -10 "$STATE/gate_r$R.log"; exit 1; }
    grep -aE "vs |delta" "$STATE/gate_r$R.log" | tail -4

    VERDICT=$(python3 - "$STATE/gate_r$R.json" <<'PY'
import json, math, sys
j = json.load(open(sys.argv[1]))
arms = j.get("arms", {})
new = arms.get("new", {})
d, se = new.get("delta_vs_baseline", 0.0), new.get("se", 0.0)
t = d / se if se else 0.0
# THE POINT ESTIMATE DECIDES. The old rule was `d <= -2 AND t <= -2`, and at 320 games
# (SE 3.6pt) t = -2 needs -7.2pt -- so every drop between -2 and -7pt was ADOPTED. Round 3
# measured -5.00pt, was adopted, and the 600-game head-to-head then put it 6.17pt (t +2.36)
# behind the round it replaced. At 600 games SE is ~2.6pt, so "d > -2" costs a ~22% false
# stop on a true zero -- and a false stop is cheap: the champion is kept.
print("ADOPT" if d > -2.0 else "REJECT")
print("delta %+.2f +- %.2f (t %+.2f)" % (d, se, t), file=sys.stderr)
PY
)
    say "round $R verdict: $VERDICT"
    if [ "$VERDICT" = "ADOPT" ]; then
        CUR=$NEW
        echo "$CUR" > "$STATE/current.txt"
        MISSES=0
    else
        MISSES=$((MISSES+1))
        say "round $R REJECTED ($MISSES in a row) -- champion stays $CUR"
        # One rejection sits within noise at this SE; two in a row from the same
        # champion means the data has stopped moving it, and more rounds are wasted GPU.
        [ "$MISSES" -ge 2 ] && { say "two rejections -- stopping for a human read"; break; }
    fi
done
say "MIRROR_CHAIN_DONE (current: $CUR)"
