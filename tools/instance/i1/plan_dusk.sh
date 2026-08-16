#!/usr/bin/env bash
# GPU side: does the MERGED plan objective hold, now that the two things that broke every
# earlier reading are gone?
#
# The merged plan probe was last measured at a gap of 0.923 and called a representation limit.
# It was measured through bf16 master weights (updates below the weight ulp round to zero) AND
# at lr 1e-4 (which, once fp32 removed the rounding floor, collapses the head to a uniform
# distribution over the menu). Both were fixed today, and at lr 1e-5 every single rule fits its
# own 300 rows to 95-100% -- boss_damaged alone stops at 79.5%. So the open question is no
# longer "is a rule representable" but "do ten of them hold AT ONCE, on rows the model has not
# seen". That is what this measures.
#
# The data has to be rebuilt: plan_r1..r4 carry the OLD prompt (with the DECK segment) and
# dusk_s1 has never seen that format.
set -u
REPO=/root/ptcg/repo; cd "$REPO"
export PYTHONPATH=cg-lib HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
SO=$REPO/data/kaggle_engine_ext/libcg_mirror.so
S1=/root/out/dusk_s1
OUT=/root/out/plan_dusk
DATA=/root/rl/plan_dusk.jsonl.gz
say() { echo "[plan $(date -u +%m-%d_%H:%M:%S)] $*"; }

TRACES=/root/traces_r4.s0.jsonl.gz,/root/traces_r4.s1.jsonl.gz,/root/traces_r4.s2.jsonl.gz,/root/traces_r5.s0.jsonl.gz,/root/rl/tr_base.jsonl.gz,/root/rl/tr_trained.jsonl.gz

# ---------------------------------------------------------------- 1. data
if [ ! -s "$DATA" ]; then
    say "building merged plan data in the dusk format"
    python3 tools/dusk_plan_data.py --traces "$TRACES" --fmt dusk --mirror-so "$SO" \
        --out "$DATA" > /root/rl/plan_dusk_build.log 2>&1 \
        || { say "STOP: build failed"; tail -5 /root/rl/plan_dusk_build.log; exit 1; }
fi
N=$(zcat "$DATA" | wc -l)
say "merged plan rows: $N"
[ "$N" -ge 2000 ] || { say "STOP: only $N rows"; exit 1; }

# ---------------------------------------------------------------- 2. probe
# Same question the old 0.923 reading answered wrongly, asked at the corrected lr. If the merged
# objective cannot memorise 300 of its own rows there is nothing for a full run to generalise.
say "=== probe: can the MERGED objective fit 300 rows at lr 1e-5? ==="
grep -aq "^PROBE OK" /root/rl/plan_dusk_probe.log 2>/dev/null || \
python3 tools/dusk_plan_train.py --data "$DATA" --model "$S1" --out /root/out/discard_planprobe \
    --probe --lr 1e-5 --epochs 10 --accum 1 > /root/rl/plan_dusk_probe.log 2>&1
PL=$(grep -a "^PROBE " /root/rl/plan_dusk_probe.log | tail -1)
say "$PL"
rm -rf /root/out/discard_planprobe
CONF=$(echo "$PL" | grep -oE "conformance [0-9.]+" | grep -oE "[0-9.]+")
Ci=$(python3 -c "print(int(float('${CONF:-0}')*10))")
if [ "$Ci" -lt 700 ]; then
    say "STOP: merged probe reaches only ${CONF}% -- the rules conflict with each other, which"
    say "      is a DIFFERENT finding from any single rule failing. Stopping for a human read."
    exit 1
fi

# ---------------------------------------------------------------- 3. train
# Held-out this time, and the L2-SP anchor ON: the probe asks about memorisation, this asks
# whether the plan survives contact with rows the model has not seen, without walking the
# checkpoint out of the basin that just scored +4.06pt.
say "=== train on all $N rows, 5% held out, lr 1e-5, anchored ==="
rm -rf "$OUT"
python3 tools/dusk_plan_train.py --data "$DATA" --model "$S1" --out "$OUT" \
    --lr 1e-5 --epochs 2 --accum 4 --l2sp 1e-3 > /root/rl/plan_dusk_train.log 2>&1 \
    || { say "STOP: train failed"; tail -8 /root/rl/plan_dusk_train.log; exit 1; }
grep -aE "\[data\]|\[eval\]|FINAL|saved" /root/rl/plan_dusk_train.log | tail -6
[ -f "$OUT/model.safetensors" ] || { say "STOP: no checkpoint saved"; exit 1; }

# ---------------------------------------------------------------- 4. gate
# Conformance is not the deliverable -- games are. Same harness and the same seeds as the gate
# that measured s1 at +4.06pt, so the two readings are comparable; s1 is carried as an arm so
# the comparison is paired rather than against a remembered number.
say "=== gate: plan vs s1 on the eleven opponents ==="
GATE=/root/loop_dusk/gate_plan
mkdir -p "$GATE"
i=0
for OPPS in "marnie_grimmsnarl,alakazam_nz,alakazam" "crustle_geco,crustle,ogerpon_mono" \
            "dudunsparce_box,cynthia_garchomp,dragapult" "mega_lucario_tr,slowking"; do
    nohup python3 -u tools/gate_protagonist.py \
        --deck dragapult_dusknoir --opp "$OPPS" --games 150 --seed $((1000 + i * 100)) \
        --arm "s1=hf:$S1@dusk" --arm "plan=hf:$OUT@dusk" \
        --out "$GATE/shard$i.json" > "$GATE/shard$i.log" 2>&1 &
    i=$((i + 1))
    sleep 45
done
wait
for k in 0 1 2 3; do
    [ -s "$GATE/shard$k.json" ] || { say "SHARD $k EMPTY"; tail -15 "$GATE/shard$k.log"; }
done
python3 -u /root/dusk_gate_pool.py "$GATE" || say "pooling failed -- shard jsons are in $GATE"
say "PLAN_DUSK_DONE"
