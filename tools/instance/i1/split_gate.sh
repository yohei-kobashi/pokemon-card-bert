#!/usr/bin/env bash
# The 7/4 split: arithmetic to the rules, judgment to the model.
#
# Runs at the SAME seed as the all-rules bundle (91000), so filter7's delta and filter11's delta
# are measured on the identical 600 (seed, seat) pairs and can be read against each other rather
# than against two different samples of noise.
#
#   RULE-BASED (7) -- forced menus and arithmetic. Every one verified against the card text:
#     spread_aim    hp <= 10*remain, else hp-10*remain <= 60; Phantom Dive's 200 hits the ACTIVE
#                   and its 6 counters hit the BENCH
#     clops_hold    Dusclops 5 counters vs Dusknoir 13, same one-prize cost
#     boss_damaged  hp <= 200 and prefer the two-prize body
#     energy_line   {R}/{P} -> the Dragapult line, {D} -> Munkidori (Fezandipiti and Meowth
#                   need no energy at all and never attack here)
#     energy_focus  does this attachment COMPLETE {R}{P} on one body
#     munki_move    free: moves damage off ours onto theirs
#     recon         free dig, declined only when the deck is nearly out (deck-count guarded)
#
#   MODEL (4) -- timing and board judgment, and the measurement agrees: on these the pilot
#   already picks correctly 93-99.5% of the times it acts, and beats chance outright on the two
#   attack rules (phantom_dive 1.74x, lock_early 1.47x). What it "gets wrong" is declining, and
#   whether to decline is exactly what the plan cannot write down.
#     evolve_line, bench_line, lock_early, phantom_dive
set -u
REPO=/root/ptcg/repo; cd "$REPO"
export PYTHONPATH=cg-lib HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export DUSK_CLOPS_HOLD=1
O=/root/loop_dusk/split; mkdir -p $O
GAMES=${GAMES:-600}
R7=spread_aim,clops_hold,boss_damaged,energy_line,energy_focus,munki_move,recon
R11=$R7,evolve_line,bench_line,lock_early,phantom_dive
say() { echo "[split $(date -u +%m-%d_%H:%M:%S)] $*"; }

say "waiting for the bundle gate"
while ! grep -aq "BUNDLE_GATE_DONE" /root/bundle_gate.log 2>/dev/null; do
    pgrep -f "bash /root/bundle_gate.sh" >/dev/null || { say "bundle gate gone without finishing"; exit 1; }
    sleep 60
done
CHAMP=$(cat /root/loop_dusk/mrl2/current.txt 2>/dev/null || true)
[ -n "$CHAMP" ] || CHAMP=/root/out/mrl_r2
say "champion $CHAMP"
pkill -f "bash /root/mirror_chain2.sh" || true
sleep 3
pkill -f "lm_mirror_log.py --model hf:" || true
sleep 5

# filter11 is re-run HERE rather than read from the bundle gate: the deck-count guard on recon
# landed between the two launches, so the bundle's 11-rule arm ran without it. Comparing across
# the two files would confound "7 vs 11 rules" with "guarded vs unguarded recon". Same process,
# same code, same seed -- then the difference is the rule set and nothing else.
say "gate: $GAMES x 4 arms | 7 vs 11 rules, seed 91000"
python3 -u tools/gate_protagonist.py \
    --deck dragapult_dusknoir --opp dragapult_dusknoir --games "$GAMES" --seed 91000 \
    --baseline cur \
    --arm "cur=hf:$CHAMP@dusk" \
    --arm "filter7=planfilter:$R7:hf:$CHAMP@dusk" \
    --arm "filter11=planfilter:$R11:hf:$CHAMP@dusk" \
    --arm "engine7=planengine:$R7:hf:$CHAMP@dusk" \
    --mirror-so "$REPO/data/kaggle_engine_ext/libcg_mirror.so" \
    --out $O/gate.json > $O/gate.log 2>&1 || say "GATE FAILED"

python3 - $O/gate.json /root/loop_dusk/bundle/gate.json <<'PY' || true
import json, sys, os
def arms(p):
    try:
        return json.load(open(p)).get("arms", {})
    except Exception:
        return {}
sp, bu = arms(sys.argv[1]), arms(sys.argv[2])
print("\n  arm                       win%   delta vs cur      t")
for lab, a, k in (("cur (champion)", sp, "cur"),
                  ("filter 11 (bundle run)", bu, "filter"),
                  ("engine 11 (bundle run)", bu, "engine"),
                  ("filter 11 (this run)", sp, "filter11"),
                  ("filter 7 rules", sp, "filter7"),
                  ("engine 7 rules", sp, "engine7")):
    v = a.get(k)
    if not v:
        continue
    d, se = v["delta_vs_baseline"], v["se"]
    print("  %-22s %6.1f%%  %+7.2f +- %.2f  %+5.2f"
          % (lab, v["win_rate"], d, se, d / se if se else 0.0))
f11 = (sp.get("filter11") or {}).get("delta_vs_baseline")
f7 = (sp.get("filter7") or {}).get("delta_vs_baseline")
if f11 is not None and f7 is not None:
    print("\n  split MINUS bundle: %+.2fpt" % (f7 - f11))
best = max([x for x in (f7, f11) if x is not None], default=None)
print("VERDICT:", "ADOPT %s" % ("the 7/4 split" if best == f7 else "all 11")
      if (best is not None and best > 2.0)
      else "deferral still does not pay -- rules belong in the REWARD, not the pilot")
PY
grep -aE "vs |^arm|^cur|^filter|^engine" $O/gate.log | tail -6

# RE-READ the champion here, not the value captured before the gate: round 6's gate was still
# running when this script started and r6_verdict.sh may have promoted a new champion since.
CHAMP=$(cat /root/loop_dusk/mrl2/current.txt 2>/dev/null || true)
[ -n "$CHAMP" ] || CHAMP=/root/out/mrl_r2
say "resuming the mirror chain (champion $CHAMP)"
cd /root
NEXT=$(grep -ac "winner:" /root/mirror_chain2.log 2>/dev/null || echo 2)
CUR="$CHAMP" TEMP=0.5 FROM=$((NEXT + 4)) ROUNDS=9 LR=2e-6 EPOCHS_FIX=0.5 L2SP=1e-2 \
    setsid nohup bash /root/mirror_chain2.sh >> /root/mirror_chain2.log 2>&1 < /dev/null &
sleep 5
say "SPLIT_GATE_DONE"
