#!/usr/bin/env bash
# Can this deck reach Phantom Dive on a human schedule? Four acceleration patterns, one control.
#
# WHAT WE ARE TRYING TO FIX. Our first Phantom Dive lands on our turn 7-8 and in 53% of games;
# the line is Dreepy -> Drakloak -> Dragapult ex plus two energy of the right colours, and the
# measurements say both halves are late: Drakloak 0.94 on our turn 2, and 25% of games never
# reach a body that can pay {R}{P} at all.
#
# THE FOUR PATTERNS, each paying with the same two Crushing Hammer slots so the comparison is
# about what was added rather than what was cut:
#   dd_candy     Rare Candy x2      -- skip the Drakloak step entirely. The general dragapult
#                                     list runs two of these and ours runs none; it is the
#                                     largest structural difference between the two decks.
#                                     Costs Recon Directive, which the guides call the reason to
#                                     evolve through Drakloak at all.
#   dd_may       May's Encouragement x2 -- two basic energy from the DISCARD onto one Stage 2,
#                                     i.e. exactly Dragapult ex. Conditional on being behind on
#                                     prizes, which against most of this field we are.
#   dd_waitress  Waitress x2        -- unconditional: look at six, attach one basic energy.
#   dd_moreen    +1 {R} +1 {P}      -- THE CONTROL. If simply running nine energy instead of
#                                     eight matches the clever cards, the clever cards are not
#                                     the answer, and every one of them costs a slot and a rule.
#
# One arm per run and a shared seed, the same shape as the Budew A/B: gate_protagonist takes one
# --deck for all its arms, so deck variants cannot share a process.
set -u
cd /root/ptcg/repo_sb
S=lethal_now,clops_hold,judge_timing,spare_ex_bench,retreat_energy,stadium_replace,search_bottom,setup_search
OPPS=marnie_grimmsnarl,alakazam_nz,dragapult,dudunsparce_box,archaludon,ogerpon_mono,mega_abomasnow_sample,ethan_hooh
for D in dragapult_dusknoir dd_candy dd_may dd_waitress dd_moreen; do
    PYTHONPATH=cg-lib:tools SB_UPTO1=1 nice -n 8 python3 -u tools/gate_protagonist.py \
        --deck "$D" --opp "$OPPS" --games 200 --seed 73000 --baseline cur \
        --opp-spec engine --mirror-so /root/ptcg/repo_sb/data/kaggle_engine_ext/libcg_mirror.so \
        --arm "cur=planfilter:$S:hf:/root/out/fld_r23b@dusk" \
        --out "/root/accel_$D.json" > "/root/accel_$D.log" 2>&1 &
done
wait
