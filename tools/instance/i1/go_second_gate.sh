#!/usr/bin/env bash
# Is taking the first turn right for THIS deck?
#
# Measured across 300 games: we were asked 150 times and answered "first" 150 times, with no rule
# involved. That is the pilot's default, not a decision anyone made. The guides call it a
# metagame trade-off -- first buys an earlier Phantom Dive, second buys Budew's item lock a turn
# sooner -- and the first half of that trade is one we are not collecting: our first Phantom Dive
# lands on our turn 7-8 and in only 53% of games.
#
# Two arms, one rule apart, on the current champion.
set -u
cd /root/ptcg/repo_sb
S=lethal_now,clops_hold,judge_timing,spare_ex_bench,retreat_energy,stadium_replace,search_bottom,setup_search
i=0
for GRP in marnie_grimmsnarl,alakazam_nz,dragapult,dudunsparce_box archaludon,ogerpon_mono,mega_abomasnow_sample,ethan_hooh; do
    i=$((i + 1))
    PYTHONPATH=cg-lib:tools SB_UPTO1=1 nice -n 8 python3 -u tools/gate_protagonist.py \
        --deck dragapult_dusknoir --opp "$GRP" --games 200 --seed 64000 --baseline first \
        --opp-spec engine --mirror-so /root/ptcg/repo_sb/data/kaggle_engine_ext/libcg_mirror.so \
        --arm "first=planfilter:$S:hf:/root/out/fld_r23b@dusk" \
        --arm "second=planfilter:$S,go_second:hf:/root/out/fld_r23b@dusk" \
        --out "/root/second_$i.json" > "/root/second_$i.log" 2>&1 &
done
wait
