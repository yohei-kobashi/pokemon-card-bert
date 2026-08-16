#!/usr/bin/env bash
# Pick the (champion, wrapper) PAIR to ship. The two facts we have were measured on different
# opponent panels and cannot be compared:
#
#   mrl2_r5b + prohibitions-only   29.6%  (n=750, the 5 decks we actually meet)
#   mrl2_r5b + R5                  25.6%  (same panel)
#   fld_r1b  + R5                  31.4%  (n=500, alakazam_nz + marnie only)
#
# fld_r1b was trained AND gated with R5 in the loop, so it is the champion best adapted to the
# wrapper that just lost by 4pt. Whether its gain survives the wrapper swap is exactly the
# question, and nothing measured so far answers it.
#
# Same panel, same seeds, three arms:
#   a  fld_r1b  + R5     -- the round-1 winner, as it was gated
#   b  fld_r1b  + proh   -- does the new champion keep its gain under the better wrapper?
#   c  mrl2_r5b + proh   -- the best pair measured so far, as the baseline to beat
set -u
REPO=/root/ptcg/repo; cd "$REPO"
export PYTHONPATH=cg-lib HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export DUSK_NEW_RULES=1 DUSK_CLOPS_HOLD=1
SO=$REPO/data/kaggle_engine_ext/libcg_mirror.so
STATE=/root/loop_dusk/champwrap; mkdir -p "$STATE"
DECK=dragapult_dusknoir
OPPS=mega_abomasnow_sample,dudunsparce_box,dragapult,archaludon,ogerpon_mono
GAMES=${GAMES:-150}
R5=lethal_now,spread_aim,clops_hold,energy_line,energy_focus
PROH=clops_hold,judge_timing,spare_ex_bench,retreat_energy,stadium_replace
say() { echo "[cw $(date -u +%m-%d_%H:%M:%S)] $*"; }

# Wait out the field chain's round-2 GATE specifically -- three concurrent gates on one card is
# where the 08-11 orphan mess came from. The branch step is CPU, so a busy GPU means a gate.
say "waiting for the field chain's round 2 to clear the GPU"
for _ in $(seq 1 180); do
    pgrep -f "[g]ate_protagonist.py --deck dragapult_dusknoir --opp alakazam_nz" >/dev/null || break
    sleep 60
done
say "starting: a=fld_r1b+R5  b=fld_r1b+proh  c=mrl2_r5b+proh, $GAMES games x 5 opponents"

python3 -u tools/gate_protagonist.py --deck "$DECK" --opp "$OPPS" --games "$GAMES" \
    --seed 96000 --baseline c --opp-spec engine \
    --arm "a=planfilter:$R5:hf:/root/out/fld_r1b@dusk" \
    --arm "b=planfilter:$PROH:hf:/root/out/fld_r1b@dusk" \
    --arm "c=planfilter:$PROH:hf:/root/out/mrl2_r5b@dusk" \
    --mirror-so "$SO" --out "$STATE/gate.json" > "$STATE/gate.log" 2>&1 \
    || { say "gate FAILED"; tail -12 "$STATE/gate.log"; exit 1; }
grep -aE "vs |delta|^arm |^a |^b |^c " "$STATE/gate.log" | tail -25
say "CHAMP_WRAP_DONE"
