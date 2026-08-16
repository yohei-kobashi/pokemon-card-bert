#!/bin/bash
# BRANCH A -- the gate found no degradation, so ask which rules the model cannot be taught.
#
# The merged probe could not answer this. It trains every rule at once and stalls at a mean gap
# of 0.923 above the target entropy; a rule that is unlearnable and a rule that simply loses its
# scope to a stronger neighbour produce the same residual there. One rule at a time, ten epochs,
# no anchor, and the only question is whether the trainer can MEMORISE it. A rule that cannot be
# memorised on its own data is not a training-budget problem -- it is either not a function of
# what the prompt shows, or the rule is mis-specified.
#
# Ten epochs rather than the merged probe's thirty because the point is separation, not the last
# fraction of loss: a rule that is representable collapses early, and one that is not does not
# arrive by epoch 30 either.
set -u
REPO=/root/ptcg/repo; cd "$REPO"
export PYTHONPATH=cg-lib HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
SO=$REPO/data/kaggle_engine_ext/libcg_mirror.so
MODEL=${MODEL:-/root/out/dusk_s1}
OUT=${OUT:-/root/rl/ruleprobe}
mkdir -p "$OUT"

# Every trace of dusknoir we have. The probe asks whether a rule is REPRESENTABLE, which does
# not depend on whose policy produced the states, so pooling sources buys rows for the rules
# that fire rarely without costing validity.
TRACES=/root/traces_r4.s0.jsonl.gz,/root/traces_r4.s1.jsonl.gz,/root/traces_r4.s2.jsonl.gz,/root/traces_r5.s0.jsonl.gz,/root/rl/tr_base.jsonl.gz,/root/rl/tr_trained.jsonl.gz

RULES=$(python3 -c "
import sys; sys.path.insert(0,'tools')
from dusk_plan import RULES
print(' '.join(RULES))")

say() { echo "[ruleprobe $(date -u +%m-%d_%H:%M:%S)] $*"; }
say "model $MODEL | rules: $RULES"
: > "$OUT/summary.txt"

for R in $RULES; do
  D=$OUT/only_$R.jsonl.gz
  if [ ! -s "$D" ]; then
    python3 tools/dusk_plan_data.py --traces "$TRACES" --only "$R" --fmt dusk \
        --mirror-so "$SO" --out "$D" > "$OUT/build_$R.log" 2>&1 \
      || { say "$R: BUILD FAILED"; echo "$R BUILD_FAILED" >> "$OUT/summary.txt"; continue; }
  fi
  N=$(zcat "$D" 2>/dev/null | wc -l)
  if [ "${N:-0}" -lt 20 ]; then
    # Fewer than 20 rows is not a failed probe, it is an absent one. Reported as such so the
    # summary cannot be read as "the model cannot learn this".
    say "$R: only $N rows -- NO DATA"
    echo "$R NO_DATA rows=$N" >> "$OUT/summary.txt"
    continue
  fi
  say "$R: $N rows -> probing"
  python3 tools/dusk_plan_train.py --data "$D" --model "$MODEL" --out "$OUT/discard_$R" \
      --probe --lr 1e-4 --epochs 10 --accum 1 > "$OUT/probe_$R.log" 2>&1
  LINE=$(grep -a "^PROBE " "$OUT/probe_$R.log" | tail -1)
  BEF=$(grep -a "conformance before" "$OUT/probe_$R.log" | tail -1 | grep -oE "[0-9.]+%" | tail -1)
  echo "$R rows=$N before=${BEF:-?} ${LINE:-PROBE_NO_OUTPUT}" >> "$OUT/summary.txt"
  say "$R: ${LINE:-no PROBE line}"
  rm -rf "$OUT/discard_$R"
done

echo
say "================ SUMMARY ================"
column -t "$OUT/summary.txt" 2>/dev/null || cat "$OUT/summary.txt"
say "logs in $OUT"
