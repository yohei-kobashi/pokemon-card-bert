#!/usr/bin/env bash
# Does the shipped planfilter wrapper stop the model attacking? Same model, same opponents,
# same seeds; the ONLY difference is the wrapper. Live games show attack taken on 8 of 56
# legal chances (14.3%) and 1.2 prizes per loss, so the question is whether the wrapper the
# 08-11 gate adopted is removing the attack from the menu -- a bug a MIRROR gate cannot see,
# because a mutual failure to attack cancels out and reads as no effect.
set -u
cd /root/ptcg/repo
OPPS=marnie_grimmsnarl,alakazam_nz,dudunsparce_box
W=planfilter:lethal_now,spread_aim,clops_hold,energy_line,energy_focus
run() {
  PYTHONPATH=cg-lib python3 tools/lm_mirror_log.py --model "$2" --deck-model engine \
    --protagonist dragapult_dusknoir --decks "$OPPS" --games 24 --seed 7000 --fmt dusk \
    --out /root/ab_$1.jsonl.gz > /root/ab_$1.log 2>&1
  echo "[$1] done rc=$?"
}
run def   "$W:hf:/root/out/mrl2_r5b"
run bare  "hf:/root/out/mrl2_r5b"
run engine engine
echo ATTACK_AB_DONE
