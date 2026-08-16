#!/usr/bin/env bash
set -u
cd /root/ptcg/repo_sb
W=lethal_now,clops_hold,judge_timing,spare_ex_bench,retreat_energy,stadium_replace,search_bottom,setup_search
for D in marnie_grimmsnarl ogerpon_mono; do
  PYTHONPATH=cg-lib:tools SB_UPTO1=1 nice -n 10 python3 -u tools/dusk_ogerpon_audit.py \
    --games 150 --opp "$D" --fmt dusk --seed 81000 \
    --mirror-so /root/ptcg/repo_sb/data/kaggle_engine_ext/libcg_mirror.so \
    --spec "planfilter:$W:hf:/root/out/fld_r11a" > /root/first_$D.log 2>&1 &
done
wait
