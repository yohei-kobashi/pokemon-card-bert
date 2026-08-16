#!/usr/bin/env bash
# The overnight rule probe answered "no rule can be learned" and the answer is not trustworthy.
# Every rule finished at a loss within a few percent of ln(mean candidate count) -- 1.6223 on
# energy_line where exp(loss)=5.06, 1.3086 on boss_damaged where exp(loss)=3.70 -- which is a
# UNIFORM distribution over the menu, and conformance fell BELOW its starting value on five of
# ten rules. A trainer that is failing to fit leaves the model where it found it; one that ends
# uniform and worse than it started has destroyed the head.
#
# The cause is almost certainly --lr 1e-4. That value was chosen while the bf16 rounding floor
# was swallowing updates and the only lever was to make each update bigger. With fp32 master
# weights the floor is gone and 3000 steps at 1e-4 on a 184M full fine-tune is simply too large;
# the same 8-row micro-test that never fit in bf16 fit in fp32 at 1e-5.
#
# Step 1 sweeps one rule across four learning rates. Step 2 re-runs all ten at whichever wins.
set -u
REPO=/root/ptcg/repo; cd "$REPO"
export PYTHONPATH=cg-lib HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
MODEL=${MODEL:-/root/out/dusk_s1}
OUT=/root/rl/ruleprobe
say() { echo "[lrsweep $(date -u +%m-%d_%H:%M:%S)] $*"; }

# energy_line: 16.5k rows, starts at 46.0% conformance, ended at 38.0%. The largest corpus of
# the rules that moved, so it has the least excuse for not fitting 300 of its own rows.
PROBE=$OUT/only_energy_line.jsonl.gz
[ -s "$PROBE" ] || { say "STOP: $PROBE missing"; exit 1; }

say "candidate counts (is the overnight loss really ln(n_cands)?)"
python3 - "$PROBE" <<'PY'
import gzip, json, math, statistics as st, sys
n = [len(json.loads(l)["cands"]) for l in gzip.open(sys.argv[1], "rt")]
print("  rows %d | mean cands %.2f | ln(mean) %.4f | overnight final loss 1.6223"
      % (len(n), st.mean(n), math.log(st.mean(n))))
PY

say "=== step 1: learning-rate sweep on energy_line ==="
BEST=""; BESTC=-1
for LR in 1e-4 3e-5 1e-5 3e-6; do
    python3 tools/dusk_plan_train.py --data "$PROBE" --model "$MODEL" \
        --out "$OUT/discard_lr$LR" --probe --lr "$LR" --epochs 10 --accum 1 \
        > "$OUT/sweep_$LR.log" 2>&1
    L=$(grep -a "^PROBE " "$OUT/sweep_$LR.log" | tail -1)
    C=$(echo "$L" | grep -oE "conformance [0-9.]+" | grep -oE "[0-9.]+")
    say "lr $LR -> ${L:-NO OUTPUT}"
    rm -rf "$OUT/discard_lr$LR"
    # Integer compare on tenths: bash has no floats and the winner is never within 0.1pt.
    Ci=$(python3 -c "print(int(float('${C:-0}')*10))")
    if [ "$Ci" -gt "$BESTC" ]; then BESTC=$Ci; BEST=$LR; fi
done
say "best lr $BEST (conformance $(python3 -c "print($BESTC/10)")%)"

# A sweep whose winner is still near chance has not found a working learning rate, and re-running
# ten rules at it would just reproduce the overnight table more slowly.
if [ "$BESTC" -lt 700 ]; then
    say "STOP: even the best lr reaches only $(python3 -c "print($BESTC/10)")% on 300 of its own"
    say "      rows. The learning rate is not the whole story -- stopping for a human read."
    exit 1
fi

say "=== step 2: all ten rules at lr $BEST ==="
RULES=$(python3 -c "
import sys; sys.path.insert(0,'tools')
from dusk_plan import RULES
print(' '.join(RULES))")
: > "$OUT/summary_lr$BEST.txt"
for R in $RULES; do
    D=$OUT/only_$R.jsonl.gz
    N=$(zcat "$D" 2>/dev/null | wc -l)
    if [ "${N:-0}" -lt 20 ]; then
        echo "$R NO_DATA rows=${N:-0}" >> "$OUT/summary_lr$BEST.txt"; continue
    fi
    python3 tools/dusk_plan_train.py --data "$D" --model "$MODEL" --out "$OUT/discard_$R" \
        --probe --lr "$BEST" --epochs 10 --accum 1 > "$OUT/probe2_$R.log" 2>&1
    LINE=$(grep -a "^PROBE " "$OUT/probe2_$R.log" | tail -1)
    BEF=$(grep -a "conformance before" "$OUT/probe2_$R.log" | tail -1 | grep -oE "[0-9.]+%" | tail -1)
    echo "$R rows=$N before=${BEF:-?} ${LINE:-PROBE_NO_OUTPUT}" >> "$OUT/summary_lr$BEST.txt"
    say "$R: ${LINE:-no PROBE line}"
    rm -rf "$OUT/discard_$R"
done
echo
say "============ SUMMARY at lr $BEST ============"
column -t "$OUT/summary_lr$BEST.txt" 2>/dev/null || cat "$OUT/summary_lr$BEST.txt"
say "LRSWEEP_DONE"
