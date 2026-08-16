#!/usr/bin/env bash
# Old-champion comeback test: fld_r41b read 53.3%/48.0% on marnie/alakazam in its era gates --
# but that era ran the OLD decklist and a smaller wrap, so the numbers are not comparable.
# Re-gate BOTH champions under todays conditions (faithful deck, current wrap), paired seeds,
# on the three hardest matchups.
set -u
while ! grep -q "PULL41B_DONE" /root/pull_r41b.log 2>/dev/null; do sleep 30; done
REPO=/root/ptcg/repo; cd "$REPO"
export PYTHONPATH=cg-lib HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export DUSK_NEW_RULES=1 DUSK_CLOPS_HOLD=1 DUSK_FRONT_DIVE=1 DUSK_BOSS_LETHAL=1 DUSK_SPIKE=1 DUSK_TIPS=1
W=$(python3 -c "import json; print(json.load(open(\"models/adapters.json\"))[\"decks\"][\"dragapult_dusknoir\"][\"wrap\"])")
nice -n 10 python3 -u tools/gate_protagonist.py \
    --deck dragapult_dusknoir --opp marnie_grimmsnarl,alakazam_nz,ogerpon_mono \
    --games "${GAMES:-150}" --seed 9500 \
    --baseline r49b \
    --arm "r49b=$W:hf:/root/out/fld_r49b@dusk" \
    --arm "r41b=$W:hf:/root/out/fld_r41b@dusk" \
    --out /root/loop_dusk/r41b_regate.json
