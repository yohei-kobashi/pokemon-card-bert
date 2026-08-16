#!/usr/bin/env bash
# Waits for ogre_wrap_ab to finish (GPU serialization), then gates lethal_line:
#   base = the shipped 18-rule wrap        vs
#   lnl  = the same wrap + lethal_line (WIN CONDITION 4: nominate the evolve/Candy that
#          unlocks a closing blast/dive -- the user-caught hole)
# Paired seeds, champion fld_r49b, the two matchups where evolve-unlocked closes appeared
# in the human games, plus marnie as a field control.
set -u
while ! grep -q "vs full" /root/ogre_wrap_ab.log 2>/dev/null; do sleep 60; done
REPO=/root/ptcg/repo; cd "$REPO"
export PYTHONPATH=cg-lib HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export DUSK_NEW_RULES=1 DUSK_CLOPS_HOLD=1 DUSK_FRONT_DIVE=1 DUSK_BOSS_LETHAL=1 DUSK_SPIKE=1 DUSK_TIPS=1
CH=/root/out/fld_r49b
FULL=lethal_now,clops_hold,judge_timing,spare_ex_bench,retreat_energy,stadium_replace,search_bottom,setup_search,front_dive,promote_dive,promote_line,lethal_boss,candy_line,noir_critical,stadium_bump,hammer_now,spike_candy,spike_race
nice -n 10 python3 -u tools/gate_protagonist.py \
    --deck dragapult_dusknoir --opp mega_abomasnow_sample,ogerpon_mono,marnie_grimmsnarl \
    --games "${GAMES:-150}" --seed 9300 \
    --baseline base \
    --arm "base=planfilter:$FULL:hf:$CH@dusk" \
    --arm "lnl=planfilter:$FULL,lethal_line:hf:$CH@dusk" \
    --out /root/loop_dusk/lnline_gate.json
