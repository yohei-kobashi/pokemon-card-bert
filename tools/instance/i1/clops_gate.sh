#!/usr/bin/env bash
# Does "hold Dusclops while Dusknoir is in hand" actually win games?
#
# The playout probe could not answer it: its unbiased statistics were flat (mean-alternative
# -0.028, sign test 62/120) and its one positive number was max-of-3 selection bias. It is also
# structurally unable to answer it -- the playout continuation is engine_v2, which never evolves
# to Dusknoir or spends the upgrade, so the benefit of holding cannot appear in the rollout.
#
# So measure the thing itself: win rate, with the rule DRIVING the decision.
#
#   cur      the champion, untouched
#   hold     planfilter -- the rule deletes the Cursed Blast option and the model picks freely
#            from the rest. NOT planrule: every non-firing option carries the same weight, so
#            the strict wrapper's argmax would pick an arbitrary one and measure that instead.
#   holdeng  engine_v2 decides those menus -- the matched control that separates "this rule is
#            right" from "anything but the model is right here"
#
# It runs BETWEEN chain rounds, not alongside them: mirror_chain2's gpu_wait demands the card be
# under 2 GiB and gives up after 30 minutes, so a co-tenant would eventually stop the chain.
set -u
REPO=/root/ptcg/repo; cd "$REPO"
export PYTHONPATH=cg-lib HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
O=/root/loop_dusk/clops; mkdir -p $O
GAMES=${GAMES:-400}
say() { echo "[clops $(date -u +%m-%d_%H:%M:%S)] $*"; }

say "waiting for mirror round 5 to reach its verdict"
while ! grep -aq "round 5 winner:" /root/mirror_chain2.log 2>/dev/null; do
    pgrep -f "bash /root/mirror_chain2.sh" >/dev/null || { say "chain died before round 5"; exit 1; }
    sleep 60
done
CHAMP=$(cat /root/loop_dusk/mrl2/current.txt 2>/dev/null || true)
[ -n "$CHAMP" ] || CHAMP=/root/out/mrl_r2
say "round 5 done; champion is $CHAMP"

say "pausing the chain so the gate owns the GPU"
pkill -f "bash /root/mirror_chain2.sh" || true
sleep 3
pkill -f "lm_mirror_log.py --model hf:" || true
sleep 5

say "gate: $GAMES paired mirror games x 3 arms, rule ENABLED via DUSK_CLOPS_HOLD=1"
DUSK_CLOPS_HOLD=1 python3 -u tools/gate_protagonist.py \
    --deck dragapult_dusknoir --opp dragapult_dusknoir --games "$GAMES" --seed 88000 \
    --baseline cur \
    --arm "cur=hf:$CHAMP@dusk" \
    --arm "hold=planfilter:clops_hold:hf:$CHAMP@dusk" \
    --arm "holdeng=planengine:clops_hold:hf:$CHAMP@dusk" \
    --mirror-so "$REPO/data/kaggle_engine_ext/libcg_mirror.so" \
    --out $O/gate.json > $O/gate.log 2>&1 || say "GATE FAILED"
grep -aE "vs |delta|^arm|^cur|^hold" $O/gate.log | tail -8

python3 - $O/gate.json <<'PY' || true
import json, sys
a = json.load(open(sys.argv[1])).get("arms", {})
for k in ("cur", "hold", "holdeng"):
    v = a.get(k)
    if v:
        d, se = v["delta_vs_baseline"], v["se"]
        print("  %-8s %5.1f%%  %+6.2f +- %.2f  t %+5.2f"
              % (k, v["win_rate"], d, se, d / se if se else 0.0))
h = (a.get("hold") or {}).get("delta_vs_baseline")
s = (a.get("hold") or {}).get("se") or 0
# One-sided: the rule earns its place only by winning. A flat result keeps it off, which is the
# same bar the four earlier deferral rules failed.
print("VERDICT:", "KEEP" if (h is not None and s and h / s > 2.0) else "DROP")
PY

say "resuming the mirror chain from round 6 (champion $CHAMP)"
cd /root
CUR="$CHAMP" TEMP=0.5 FROM=6 ROUNDS=9 LR=2e-6 EPOCHS_FIX=0.5 L2SP=1e-2 \
    setsid nohup bash /root/mirror_chain2.sh >> /root/mirror_chain2.log 2>&1 < /dev/null &
sleep 5
say "CLOPS_GATE_DONE"
