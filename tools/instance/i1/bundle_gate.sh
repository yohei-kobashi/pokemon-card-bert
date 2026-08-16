#!/usr/bin/env bash
# ALL rules at once, against no rules.
#
# Rule-by-rule A/B has been run four times and every arm landed inside the noise (strict -0.83,
# filter -2.50, engine +1.88 at SE ~2pt). Each rule is a small effect on a subset of decisions;
# at 600 games the gate cannot resolve one. The user's call: bundle everything worth adopting
# and ask the only question that has a chance of answering itself -- rules or no rules.
#
#   cur      the champion, untouched
#   filter   the plan deletes non-conformant options, the model ranks the survivors
#   engine   engine_v2 decides every menu a rule fires on -- the matched control that separates
#            "these rules are right" from "anything but the model is right there"
#
# spread_aim is the CORRECTED one: it used to grade bench placements against Phantom Dive's 200,
# which only ever hits the Active. On the same 800 games the fix moved the pilot's measured
# execution from 14.1% (0.51x chance) to 46.6% (1.15x) without the pilot changing at all.
set -u
REPO=/root/ptcg/repo; cd "$REPO"
export PYTHONPATH=cg-lib HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export DUSK_CLOPS_HOLD=1          # the bundle includes the hold rule
O=/root/loop_dusk/bundle; mkdir -p $O
GAMES=${GAMES:-600}
RULES=spread_aim,clops_hold,boss_damaged,energy_line,energy_focus,munki_move,recon,evolve_line,bench_line,lock_early,phantom_dive
say() { echo "[bundle $(date -u +%m-%d_%H:%M:%S)] $*"; }

say "waiting for the chain to finish its current round"
while pgrep -f "bash /root/mirror_chain2.sh" >/dev/null; do
    if grep -aq "round 6 winner:" /root/mirror_chain2.log 2>/dev/null; then break; fi
    sleep 60
done
CHAMP=$(cat /root/loop_dusk/mrl2/current.txt 2>/dev/null || true)
[ -n "$CHAMP" ] || CHAMP=/root/out/mrl_r2
say "champion $CHAMP"
pkill -f "bash /root/mirror_chain2.sh" || true
sleep 3
pkill -f "lm_mirror_log.py --model hf:" || true
sleep 5

say "gate: $GAMES paired mirror games x 3 arms | 11 rules"
python3 -u tools/gate_protagonist.py \
    --deck dragapult_dusknoir --opp dragapult_dusknoir --games "$GAMES" --seed 91000 \
    --baseline cur \
    --arm "cur=hf:$CHAMP@dusk" \
    --arm "filter=planfilter:$RULES:hf:$CHAMP@dusk" \
    --arm "engine=planengine:$RULES:hf:$CHAMP@dusk" \
    --mirror-so "$REPO/data/kaggle_engine_ext/libcg_mirror.so" \
    --out $O/gate.json > $O/gate.log 2>&1 || say "GATE FAILED"
grep -aE "vs |delta|^arm|^cur|^filter|^engine" $O/gate.log | tail -8

python3 - $O/gate.json <<'PY' || true
import json, sys
a = json.load(open(sys.argv[1])).get("arms", {})
for k in ("cur", "filter", "engine"):
    v = a.get(k)
    if v:
        d, se = v["delta_vs_baseline"], v["se"]
        print("  %-8s %5.1f%%  %+6.2f +- %.2f  t %+5.2f"
              % (k, v["win_rate"], d, se, d / se if se else 0.0))
f = (a.get("filter") or {}).get("delta_vs_baseline")
print("VERDICT:", "ADOPT the bundle" if (f is not None and f > 2.0)
      else "rule deferral does not pay -- keep the rules in the REWARD only")
PY

say "resuming the mirror chain (champion $CHAMP)"
cd /root
NEXT=$(grep -ac "winner:" /root/mirror_chain2.log 2>/dev/null || echo 5)
CUR="$CHAMP" TEMP=0.5 FROM=$((NEXT + 4)) ROUNDS=9 LR=2e-6 EPOCHS_FIX=0.5 L2SP=1e-2 \
    setsid nohup bash /root/mirror_chain2.sh >> /root/mirror_chain2.log 2>&1 < /dev/null &
sleep 5
say "BUNDLE_GATE_DONE"
