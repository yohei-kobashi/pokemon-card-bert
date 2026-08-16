#!/usr/bin/env bash
# Baseline -> probe -> train -> evaluate, all on the EXECUTION table. No win rate anywhere.
#
# The baseline has to come from d41_r8 ITSELF. The table measured so far was replayed from
# traces_r4, which the 4B produced -- a different model's execution is not this model's
# starting point, and comparing against it would credit or blame the wrong policy.
set -u
say() { echo "[plan $(date -u +%m-%d_%H:%M:%S)] $*"; }
REPO=/root/ptcg/repo; cd "$REPO"
export PYTHONPATH=cg-lib HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
SO=$REPO/data/kaggle_engine_ext/libcg_mirror.so
OPPS=$(python3 -c "
import sys; sys.path.insert(0,'tools'); import rl_config
print(','.join(d for d in rl_config.STAGE_C_TARGETS if d!='dragapult_dusknoir'))")

trace_and_measure() {   # $1 model  $2 tag  $3 games
  say "traces from $1 ($3 games/opponent)"
  python3 tools/lm_mirror_log.py --model "hf:$1" --protagonist dragapult_dusknoir \
    --decks "$OPPS" --games "$3" --seed 990000 --mirror-so "$SO" \
    --out /root/rl/lm_$2.jsonl.gz --trace-out /root/rl/tr_$2.jsonl.gz > /root/rl/tr_$2.log 2>&1 \
    || { say "trace generation FAILED for $2"; tail -3 /root/rl/tr_$2.log; return 1; }
  say "execution table: $2"
  python3 tools/dusk_plan.py --traces /root/rl/tr_$2.jsonl.gz --mirror-so "$SO" \
    2>&1 | tee /root/rl/exec_$2.txt | tail -12
}

trace_and_measure /root/out/d41_r8 base 12 || exit 1

say "probe 30 epochs (floor 0.278)"
python3 tools/dusk_plan_train.py --data /root/rl/plan_r4.jsonl.gz --model /root/out/d41_r8 \
  --out /root/out/plan_probe --probe --lr 1e-4 --epochs 30 --accum 1 2>&1 \
  | grep -aE "^\[probe\]|^FINAL|^PROBE"

say "train"
python3 tools/dusk_plan_train.py --data /root/rl/plan_r4.jsonl.gz --model /root/out/d41_r8 \
  --out /root/out/plan_r1 --lr 5e-5 --epochs 1 --accum 1 --l2sp 1e-3 \
  > /root/rl/plan_train1.log 2>&1 || { say "TRAIN FAILED"; tail -5 /root/rl/plan_train1.log; exit 1; }
grep -aE "\[data\]|\[l2sp\]|\[eval\]|FINAL|saved" /root/rl/plan_train1.log | tail -6

trace_and_measure /root/out/plan_r1 trained 12 || exit 1

say "BEFORE vs AFTER"
python3 - <<'PY'
import re
def load(p):
    out = {}
    for ln in open(p):
        m = re.match(r"(\w+)\s+(\d+|-)\s+(\d+)\s+([\d.]+)%", ln.strip())
        if m:
            out[m.group(1)] = (int(m.group(2)) if m.group(2) != "-" else 0,
                               int(m.group(3)), float(m.group(4)))
    return out
b, a = load("/root/rl/exec_base.txt"), load("/root/rl/exec_trained.txt")
print("%-14s %10s %10s %9s" % ("rule", "d41_r8", "trained", "delta"))
for k in sorted(set(b) | set(a), key=lambda k: -(a.get(k, (0, 0, 0))[2] - b.get(k, (0, 0, 0))[2])):
    x, y = b.get(k, (0, 0, float("nan"))), a.get(k, (0, 0, float("nan")))
    print("%-14s %9.1f%% %9.1f%% %+8.1f   (n %d -> %d)" % (k, x[2], y[2], y[2] - x[2], x[1], y[1]))
PY
say PLAN_PIPELINE_DONE
