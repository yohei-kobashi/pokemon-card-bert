#!/usr/bin/env bash
# instance1, overnight: (a) keep the RL chain alive, (b) A/B the Judge -> Budew swap,
# (c) collect instance2's 4B verdict when it lands. Nothing here needs a human.
#
# WHY THE A/B RUNS UNCONDITIONALLY. The decision rule is "swap if the 4B did not improve
# either", but that governs whether to ADOPT the swap, not whether to MEASURE it. Measuring
# costs one core (gate_protagonist is a single process; field_chain's own gate is too), and
# having both numbers in the morning is strictly better than having one and a coin flip.
#
# WHY IT DOES NOT TOUCH decks/dragapult_dusknoir.csv. The live chain is training a champion on
# the current 60 cards and cannot be reviewed until morning; editing the deck under it would
# invalidate the round in flight and there would be no one to notice. The swap lives in a
# SEPARATE deck file, so applying it later is a one-line change made with data in hand.
set -u
REPO=/root/ptcg/repo; cd "$REPO"
LOG=/root/overnight.log
export PYTHONPATH=cg-lib:tools HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export DUSK_NEW_RULES=1 DUSK_CLOPS_HOLD=1
SO=$REPO/data/kaggle_engine_ext/libcg_mirror.so
OPPS=marnie_grimmsnarl,alakazam_nz,dragapult,dudunsparce_box,archaludon,ogerpon_mono,mega_abomasnow_sample,ethan_hooh
GAMES=${GAMES:-250}
SEED=${SEED:-97000}
PFX="planfilter:lethal_now,clops_hold,judge_timing,spare_ex_bench,retreat_energy,stadium_replace:"
say() { echo "[night $(date -u +%m-%d_%H:%M:%S)] $*" >> "$LOG"; }

CHAMP=$(python3 - <<'PY'
import json, os
r = json.load(open("/root/ptcg/repo/models/adapters.json"))
t = (r["decks"]["dragapult_dusknoir"]["target"] or "").partition(":")[2]
print(t if t.startswith("/") else os.path.join("/root/out", t))
PY
)
say "champion for the A/B: $CHAMP (frozen for both arms)"
[ -f "$CHAMP/model.safetensors" ] || { say "STOP: champion has no weights"; exit 1; }

# ---- Judge -> Budew, measured on the shipping pilot -------------------------------------
# Two runs rather than two arms: gate_protagonist takes ONE --deck for all its arms, so a deck
# change cannot be an arm. Same seed, same opponents, same pilot, same wrapper -- the decklist
# is the only difference. Note this is NOT paired the way two arms inside one gate are: a
# different 60 cards shuffles differently, so read it with the ~4pt SE of two 1200-game runs,
# not with the +-1.3pt of an in-gate delta.
for D in dragapult_dusknoir dragapult_dusknoir_budew2; do
    OUT=/root/night_${D}.json
    if [ -s "$OUT" ]; then say "$D already measured"; continue; fi
    say "gate: $D vs 8 opponents, $GAMES games each (seed $SEED)"
    nice -n 10 python3 -u tools/gate_protagonist.py --deck "$D" --opp "$OPPS" \
        --games "$GAMES" --seed "$SEED" --baseline cur --opp-spec engine \
        --arm "cur=${PFX}hf:$CHAMP@dusk" --mirror-so "$SO" --out "$OUT" \
        > /root/night_${D}.log 2>&1 || say "$D gate FAILED (see /root/night_${D}.log)"
    say "$D done"
done

python3 - <<'PY' >> "$LOG" 2>&1
import json, os
a, b = "/root/night_dragapult_dusknoir.json", "/root/night_dragapult_dusknoir_budew2.json"
if not (os.path.exists(a) and os.path.exists(b)):
    raise SystemExit("one of the two runs is missing")
A, B = json.load(open(a)), json.load(open(b))
def cells(j):
    return {k.split("|", 1)[1]: (v["win"], v["games"]) for k, v in j["cells"].items()
            if k.startswith("cur|")}
ca, cb = cells(A), cells(B)
print("\n=========== Judge 1  ->  Budew 2 ===========")
print("%-24s %8s %8s %8s" % ("opponent", "base", "budew2", "delta"))
for o in sorted(ca, key=lambda x: ca[x][0] / max(1, ca[x][1])):
    wa = 100.0 * ca[o][0] / max(1, ca[o][1])
    wb = 100.0 * cb[o][0] / max(1, cb[o][1])
    print("%-24s %7.1f%% %7.1f%% %+7.1f" % (o, wa, wb, wb - wa))
ta = 100.0 * sum(w for w, _g in ca.values()) / max(1, sum(g for _w, g in ca.values()))
tb = 100.0 * sum(w for w, _g in cb.values()) / max(1, sum(g for _w, g in cb.values()))
n = sum(g for _w, g in ca.values())
se = (2 * (ta / 100 * (1 - ta / 100) / n)) ** 0.5 * 100
print("%-24s %7.1f%% %7.1f%% %+7.1f   (SE about %.1fpt, unpaired)" % ("TOTAL", ta, tb, tb - ta, se))
print("Adopt only if the total is up by more than that SE AND ogerpon_mono is not the only cell moving.")
PY
say "A/B written to $LOG"

# ---- instance2's 4B verdict, whenever it lands -------------------------------------------
for _ in $(seq 1 90); do
    V=$(ssh -i /root/.ssh/id_i2 -o IdentitiesOnly=yes -o StrictHostKeyChecking=no \
        -o BatchMode=yes -o ConnectTimeout=20 -p 19839 root@175.155.64.145 \
        'sed -n "/RESULT/,$p" /root/dusk_vs_oger.log 2>/dev/null' 2>/dev/null)
    if [ -n "$V" ]; then
        say "instance2 4B verdict:"; echo "$V" >> "$LOG"; break
    fi
    sleep 300
done
say "OVERNIGHT_DONE"
