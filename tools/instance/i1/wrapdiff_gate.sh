#!/usr/bin/env bash
# Morning-decision evidence: does the NEW wrap (dusk_v4: +hammer_spare +lethal_line +draw_cap)
# beat the SHIPPED dusk_v3 wrap (18 rules) on the four human-sparred matchups? Starts at loop
# stop (23:00Z) when the GPU frees up. Champion held fixed at whatever the registry says.
set -u
until [ "$(date -u +%s)" -ge "$(date -u -d 2026-08-16T23:02:00Z +%s)" ]; do sleep 60; done
REPO=/root/ptcg/repo; cd "$REPO"
export PYTHONPATH=cg-lib HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export DUSK_NEW_RULES=1 DUSK_CLOPS_HOLD=1 DUSK_FRONT_DIVE=1 DUSK_BOSS_LETHAL=1 DUSK_SPIKE=1 DUSK_TIPS=1
CH=/root/out/fld_r49b
V3=lethal_now,clops_hold,judge_timing,spare_ex_bench,retreat_energy,stadium_replace,search_bottom,setup_search,front_dive,promote_dive,promote_line,lethal_boss,candy_line,noir_critical,stadium_bump,hammer_now,spike_candy,spike_race
V4=$V3,hammer_spare,lethal_line,draw_cap
nice -n 5 python3 -u tools/gate_protagonist.py \
    --deck dragapult_dusknoir --opp marnie_grimmsnarl,alakazam_nz,ogerpon_mono,mega_abomasnow_sample \
    --games 100 --seed 9700 \
    --baseline v3wrap \
    --arm "v3wrap=planfilter:$V3:hf:$CH@dusk" \
    --arm "v4wrap=planfilter:$V4:hf:$CH@dusk" \
    --out /root/loop_dusk/wrapdiff_gate.json
