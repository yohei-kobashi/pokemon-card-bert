#!/usr/bin/env bash
# Does the positive-rule wrap HELP or HURT vs ogerpon_mono?
# The users 16 play_server games vs ogerpon (9-7, vs our stacks 12-21%) violate the positive
# rules wholesale: dive forced (2/33 turns conformed), armed-pult promotion (human fronts
# Munkidori), clops_hold (human fires Dusclops blast with Dusknoir in hand on purpose).
# Four arms, paired (seed, seat), champion fld_r49b, ogerpon_mono only:
#   full   the shipped 18-rule wrap
#   lite   lethal_now + prohibitions only (all other positive rules off)
#   noclp  lite minus clops_hold (the prohibition the human deliberately violates)
#   bare   no wrap at all
set -u
REPO=/root/ptcg/repo; cd "$REPO"
export PYTHONPATH=cg-lib HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export DUSK_NEW_RULES=1 DUSK_CLOPS_HOLD=1 DUSK_FRONT_DIVE=1 DUSK_BOSS_LETHAL=1 DUSK_SPIKE=1 DUSK_TIPS=1
CH=/root/out/fld_r49b
FULL=lethal_now,clops_hold,judge_timing,spare_ex_bench,retreat_energy,stadium_replace,search_bottom,setup_search,front_dive,promote_dive,promote_line,lethal_boss,candy_line,noir_critical,stadium_bump,hammer_now,spike_candy,spike_race
LITE=lethal_now,clops_hold,judge_timing,spare_ex_bench,retreat_energy,stadium_replace,search_bottom,noir_critical
NOCLP=lethal_now,judge_timing,spare_ex_bench,retreat_energy,stadium_replace,search_bottom,noir_critical
nice -n 10 python3 -u tools/gate_protagonist.py \
    --deck dragapult_dusknoir --opp ogerpon_mono --games "${GAMES:-200}" --seed 9100 \
    --baseline full \
    --arm "full=planfilter:$FULL:hf:$CH@dusk" \
    --arm "lite=planfilter:$LITE:hf:$CH@dusk" \
    --arm "noclp=planfilter:$NOCLP:hf:$CH@dusk" \
    --arm "bare=hf:$CH@dusk" \
    --out /root/loop_dusk/ogre_wrap_ab.json
