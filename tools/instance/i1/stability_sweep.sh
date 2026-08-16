#!/usr/bin/env bash
# Find a training setting that is REPRODUCIBLE, then resume the mirror chain with it.
#
# WHY. r3diag settled the round-2 -> round-3 question and the answer was not a round-3 defect.
# Two checkpoints built from the SAME pairs, the SAME parent and the SAME recipe, differing only
# in the ORDER OF THE ROWS -- a change that cannot alter the objective -- came out 26.00pt apart
# (r3 55.3%, r3s 29.3%). Round 4 then produced -36.67 and -12.83 from the same champion. So the
# training step is the variance source, the gate is not, and every adoption decision the v1 chain
# ever made was a draw from that spread.
#
# WHAT THIS MEASURES. Four settings x two row orders, all from the champion, all on round 4's
# pairs. The number that decides is the SPREAD between the two orders: a setting whose two draws
# land far apart cannot be learned from, whatever its mean. Only then does the mean matter.
#
# The gate also re-measures the champion on a FRESH seed set. Both previous 600-game readings of
# mrl_r2 used seed 33000; if its 61.5% is partly a property of those particular shuffles, this is
# where that shows up.
set -u
REPO=/root/ptcg/repo; cd "$REPO"
export PYTHONPATH=cg-lib HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
O=/root/loop_dusk/sweep; mkdir -p $O
CUR=/root/out/mrl_r2
PAIRS=/root/mrl2_pairs4.jsonl.gz
GATE_GAMES=${GATE_GAMES:-300}
say() { echo "[sweep $(date -u +%m-%d_%H:%M:%S)] $*"; }

gpu_wait() {
    local u
    for _ in $(seq 1 60); do
        u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
        [ "$u" -le 2000 ] && return 0
        sleep 30
    done
    say "STOP: GPU held ${u} MiB for 30 min"; exit 1
}

say "rows from $PAIRS (beta 0, temp 0.5), two orders"
python3 /root/mrl_convert.py --pairs $PAIRS --out $O/rows_A.jsonl.gz --beta 0.0 --temp 0.5 \
    | tee $O/convert.log
python3 - <<'PY'
import gzip, random
src = "/root/loop_dusk/sweep/rows_A.jsonl.gz"
rows = list(gzip.open(src, "rt"))
random.Random(11).shuffle(rows)
with gzip.open("/root/loop_dusk/sweep/rows_A.jsonl.gz", "wt") as f:
    f.writelines(rows)
random.Random(22).shuffle(rows)
with gzip.open("/root/loop_dusk/sweep/rows_B.jsonl.gz", "wt") as f:
    f.writelines(rows)
print("two row orders written (%d rows each)" % len(rows))
PY

# name  lr     epochs  l2sp
SETTINGS="s1:1e-5:2.0:1e-3 s2:2e-6:2.0:1e-3 s3:1e-5:0.25:1e-3 s4:2e-6:0.5:1e-2"
ARMS="--arm cur=hf:$CUR@dusk"
for S in $SETTINGS; do
    NAME=$(echo $S | cut -d: -f1); LR=$(echo $S | cut -d: -f2)
    EP=$(echo $S | cut -d: -f3);   L2=$(echo $S | cut -d: -f4)
    for ORD in A B; do
        OUT=/root/out/sw_${NAME}$ORD
        if [ ! -f "$OUT/model.safetensors" ]; then
            gpu_wait
            say "train $NAME$ORD: lr $LR epochs $EP l2sp $L2"
            python3 tools/dusk_plan_train.py --data $O/rows_$ORD.jsonl.gz --model "$CUR" \
                --out "$OUT" --lr "$LR" --epochs "$EP" --accum 4 --l2sp "$L2" \
                > $O/train_$NAME$ORD.log 2>&1 \
                || { say "train $NAME$ORD FAILED"; tail -4 $O/train_$NAME$ORD.log; continue; }
            grep -aE "FINAL|\[eval\]" $O/train_$NAME$ORD.log | tail -2
        fi
        [ -f "$OUT/model.safetensors" ] && ARMS="$ARMS --arm $NAME$ORD=hf:$OUT@dusk"
    done
done

gpu_wait
say "gate: $GATE_GAMES paired mirror games, FRESH seed set (77000), arms: $ARMS"
python3 -u tools/gate_protagonist.py --deck dragapult_dusknoir --opp dragapult_dusknoir \
    --games "$GATE_GAMES" --seed 77000 --baseline cur $ARMS \
    --out $O/gate.json 2>&1 | tail -20

CHOICE=$(python3 - $O/gate.json <<'PY'
import json, sys
a = json.load(open(sys.argv[1])).get("arms", {})
rows = []
for name, lr, ep, l2 in (("s1", "1e-5", "2.0", "1e-3"), ("s2", "2e-6", "2.0", "1e-3"),
                         ("s3", "1e-5", "0.25", "1e-3"), ("s4", "2e-6", "0.5", "1e-2")):
    va, vb = a.get(name + "A"), a.get(name + "B")
    if not va or not vb:
        continue
    da, db = va["delta_vs_baseline"], vb["delta_vs_baseline"]
    rows.append((abs(da - db), (da + db) / 2.0, name, lr, ep, l2, da, db))
for sp, mn, n, lr, ep, l2, da, db in sorted(rows):
    print("  %-3s lr %-5s ep %-5s l2sp %-5s  A %+7.2f  B %+7.2f  spread %6.2f  mean %+7.2f"
          % (n, lr, ep, l2, da, db, sp, mn), file=sys.stderr)
# STABILITY FIRST. A setting whose two row orders land more than 5pt apart is not a setting, it
# is a lottery; its mean says nothing about what the next round would draw. Among the stable
# ones, take the best mean. If none is stable, say so and let a human decide.
stable = [r for r in rows if r[0] <= 5.0]
if not stable:
    print("NONE")
else:
    best = max(stable, key=lambda r: r[1])
    print("%s %s %s %.2f %.2f" % (best[3], best[4], best[5], best[1], best[0]))
PY
)
echo "$CHOICE" > $O/choice.txt
say "choice: $CHOICE"

if [ "$(echo $CHOICE | cut -d' ' -f1)" = "NONE" ]; then
    say "NO STABLE SETTING -- not resuming the chain. Every candidate's two row orders landed"
    say "more than 5pt apart, so another round would be another coin flip. Human read needed."
    exit 0
fi
LR=$(echo $CHOICE | cut -d' ' -f1); EP=$(echo $CHOICE | cut -d' ' -f2); L2=$(echo $CHOICE | cut -d' ' -f3)
say "resuming the mirror chain from round 5 with lr $LR epochs $EP l2sp $L2"
cd /root
CUR=$CUR TEMP=0.5 FROM=5 ROUNDS=9 LR="$LR" EPOCHS_FIX="$EP" L2SP="$L2" \
    setsid nohup bash /root/mirror_chain2.sh >> /root/mirror_chain2.log 2>&1 < /dev/null &
sleep 5
say "SWEEP_DONE -- chain resumed"
