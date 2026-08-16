#!/usr/bin/env bash
# Why did round 3 lose 6.17pt to round 2?
#
# The inputs are indistinguishable: pairs 1762 vs 1902, |Q margin| p50 0.348 vs 0.339,
# model_right/wrong 53%/52%, steps 3348 vs 3614, and BOTH trainings end at -log P = 0.693 = ln 2.
# So there is no round-3-specific defect visible in the data. Two candidate explanations, and
# this script separates them by rebuilding round 3 three more ways from the SAME pairs and the
# SAME champion:
#
#   r3      the checkpoint that lost      beta 0.3, temp 0.5, original row order
#   r3s     same recipe, rows SHUFFLED    -> isolates pure optimisation-path variance
#   r3q     beta 0 (labels from Q alone)  -> beta blended against rww=rwl=0 (the spawn bug), so
#                                            0.3 was never rule conformance; it was 30% label
#                                            smoothing toward uniform. This removes it.
#   r3sharp beta 0, temp 0.25             -> the same labels, twice as sharp
#
# If r3s lands as far from r3 as r3 is from r2, the chain is a random walk and the answer is a
# champion-vs-challenger design, not a fix. If r3q/r3sharp beat r3, the labels were too soft.
set -u
REPO=/root/ptcg/repo; cd "$REPO"
export PYTHONPATH=cg-lib HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
O=/root/loop_dusk/r3diag; mkdir -p $O
CUR=/root/out/mrl_r2
say() { echo "[r3diag $(date -u +%m-%d_%H:%M:%S)] $*"; }

say "rebuilding round 3's rows three ways from /root/mrl_pairs3.jsonl.gz"
python3 /root/mrl_convert.py --pairs /root/mrl_pairs3.jsonl.gz --out $O/rows_q.jsonl.gz \
    --beta 0.0 --temp 0.5 | tee $O/convert_q.log
python3 /root/mrl_convert.py --pairs /root/mrl_pairs3.jsonl.gz --out $O/rows_sharp.jsonl.gz \
    --beta 0.0 --temp 0.25 | tee $O/convert_sharp.log
python3 /root/mrl_convert.py --pairs /root/mrl_pairs3.jsonl.gz --out $O/rows_s.jsonl.gz \
    --beta 0.3 --temp 0.5 | tee $O/convert_s.log

# Reshuffle for the variance arm. dusk_plan_train has no --seed, so the row ORDER is the only
# knob that moves the optimisation path without changing the objective at all.
python3 - <<'PY'
import gzip, random
rows = list(gzip.open("/root/loop_dusk/r3diag/rows_s.jsonl.gz", "rt"))
random.Random(20260811).shuffle(rows)
with gzip.open("/root/loop_dusk/r3diag/rows_s.jsonl.gz", "wt") as f:
    f.writelines(rows)
print("shuffled %d rows" % len(rows))
PY

for V in q sharp s; do
    say "train $V from $CUR"
    python3 tools/dusk_plan_train.py --data $O/rows_$V.jsonl.gz --model "$CUR" \
        --out /root/out/mrl_r3$V --lr 1e-5 --epochs 2.0 --accum 4 --l2sp 1e-3 \
        > $O/train_$V.log 2>&1 || { say "train $V FAILED"; tail -5 $O/train_$V.log; exit 1; }
    grep -aE "FINAL|\[eval\]" $O/train_$V.log | tail -2
done

say "gate: 5 arms x 600 paired mirror games vs engine_v2"
python3 -u tools/gate_protagonist.py --deck dragapult_dusknoir --opp dragapult_dusknoir \
    --games 600 --seed 33000 --baseline cur \
    --arm "cur=hf:$CUR@dusk" \
    --arm "r3=hf:/root/out/mrl_r3@dusk" \
    --arm "r3s=hf:/root/out/mrl_r3s@dusk" \
    --arm "r3q=hf:/root/out/mrl_r3q@dusk" \
    --arm "r3sharp=hf:/root/out/mrl_r3sharp@dusk" \
    --out $O/gate.json 2>&1 | tail -20
say "R3DIAG_DONE"
