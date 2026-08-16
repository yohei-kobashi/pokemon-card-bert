#!/usr/bin/env bash
# 2x2 factorial over the two holes the setup rules did NOT close:
#   en  = energy_line,energy_focus   -> the 25% of games with no payable Phantom Dive
#   pd  = phantom_dive               -> the 1.82 turns between payable and firing
# `both` is here because sb -> ss already showed these rules compose (+4.10 -> +6.55); main
# effects alone would have missed that.
set -u
cd /root/ptcg/repo_sb
S=lethal_now,clops_hold,judge_timing,spare_ex_bench,retreat_energy,stadium_replace,search_bottom,setup_search
i=0
for GRP in marnie_grimmsnarl,alakazam_nz,dragapult,dudunsparce_box archaludon,ogerpon_mono,mega_abomasnow_sample,ethan_hooh; do
  i=$((i+1))
  PYTHONPATH=cg-lib:tools SB_UPTO1=1 nice -n 8 python3 -u tools/gate_protagonist.py \
    --deck dragapult_dusknoir --opp "$GRP" --games 150 --seed 52000 --baseline ss \
    --opp-spec engine --mirror-so /root/ptcg/repo_sb/data/kaggle_engine_ext/libcg_mirror.so \
    --arm "ss=planfilter:$S:hf:/root/out/fld_r11a@dusk" \
    --arm "en=planfilter:$S,energy_line,energy_focus:hf:/root/out/fld_r11a@dusk" \
    --arm "pd=planfilter:$S,phantom_dive:hf:/root/out/fld_r11a@dusk" \
    --arm "both=planfilter:$S,energy_line,energy_focus,phantom_dive:hf:/root/out/fld_r11a@dusk" \
    --out /root/hole_$i.json > /root/hole_$i.log 2>&1 &
done
wait
