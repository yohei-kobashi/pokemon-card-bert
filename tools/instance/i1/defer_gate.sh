#!/usr/bin/env bash
# WHICH RULES SHOULD THE MODEL STOP DECIDING?
#
# Audit of the 19 live games of submission 55445834 (tools/live_rule_audit.py), model-owned rules:
#
#   retreat_energy  305 fired   99.0% obeyed vs 81.6% chance   +17.4
#   evolve_line      91          81.3%        vs 27.3%         +54.1
#   recon            70          72.9%        vs 19.6%         +53.2
#   bench_line       45          60.0%        vs 22.6%         +37.4
#   judge_timing     26          76.9%        vs 83.8%          -6.8   <-- BELOW CHANCE
#   spare_ex_bench   13          76.9%        vs 85.1%          -8.2   <-- BELOW CHANCE
#
# The two below chance are both PROHIBITIONS ("do not Judge while...", "do not bench a body worth
# the prizes they need"). That shape is what makes them cheap to hand over: filtering to
# (everything except the forbidden option) removes one move and leaves the model every other
# choice, unlike a positive rule, where filtering to it FORCES the action and takes the turn.
# Judge is also a card added to the list on 08-11, after the model's last training data -- it has
# never seen the decision it is being scored on.
#
# Arms, all on the same champion:
#   r5    what shipped: lethal_now,spread_aim,clops_hold,energy_line,energy_focus
#   r9    r5 + the four prohibitions (judge_timing, spare_ex_bench, retreat_energy,
#         stadium_replace) -- the "hand over everything phrased as a prohibition" arm
#   proh  prohibitions ONLY -- separates "the prohibitions help" from "the positive rules help",
#         which r9-vs-r5 alone cannot
#
# OPPONENTS ARE THE ONES WE ACTUALLY MET. The leaderboard's top-2 (alakazam_nz, marnie) appeared
# in ZERO of the 19 live games; at this rating the field is mega_abomasnow, dudunsparce, dragapult,
# archaludon and ogerpon. Gating on the top-500 distribution measures a bracket we are not in.
set -u
REPO=/root/ptcg/repo; cd "$REPO"
export PYTHONPATH=cg-lib HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
# The flags the FIXED bundle now sets. Without them six rules keep their names and lose their
# behaviour, which is how two of r5's five rules shipped inert.
export DUSK_NEW_RULES=1 DUSK_CLOPS_HOLD=1
SO=$REPO/data/kaggle_engine_ext/libcg_mirror.so
STATE=/root/loop_dusk/defer; mkdir -p "$STATE"
DECK=dragapult_dusknoir
CUR=${CUR:-/root/out/mrl2_r5b}
OPPS=${OPPS:-mega_abomasnow_sample,dudunsparce_box,dragapult,archaludon,ogerpon_mono}
GAMES=${GAMES:-150}
R5=lethal_now,spread_aim,clops_hold,energy_line,energy_focus
PROH=clops_hold,judge_timing,spare_ex_bench,retreat_energy,stadium_replace
R9=lethal_now,spread_aim,clops_hold,energy_line,energy_focus,judge_timing,spare_ex_bench,retreat_energy,stadium_replace
say() { echo "[defer $(date -u +%m-%d_%H:%M:%S)] $*"; }

# NO GPU WAIT. The field chain runs its rounds back to back, so "wait for an idle GPU" meant
# waiting for a gap that never opens -- the gate sat idle through round 1 and would have sat
# through every later one. DeBERTa-v3-base is ~1.5 GiB and the field gate peaks near 7.6 of the
# card's 24, so the two fit together; they trade throughput, not correctness, and with the
# deadline on 08-16 a slower answer to both beats a fast answer to one.
say "GPU free; arms r5 / r9 / proh vs $OPPS, $GAMES games per (arm, opponent)"

python3 -u tools/gate_protagonist.py --deck "$DECK" --opp "$OPPS" --games "$GAMES" \
    --seed 93000 --baseline r5 --opp-spec engine \
    --arm "r5=planfilter:$R5:hf:$CUR@dusk" \
    --arm "r9=planfilter:$R9:hf:$CUR@dusk" \
    --arm "proh=planfilter:$PROH:hf:$CUR@dusk" \
    --mirror-so "$SO" --out "$STATE/gate.json" > "$STATE/gate.log" 2>&1 \
    || { say "gate FAILED"; tail -12 "$STATE/gate.log"; exit 1; }
grep -aE "vs |delta|^arm |^r5 |^r9 |^proh " "$STATE/gate.log" | tail -25
say "DEFER_GATE_DONE"
