#!/bin/bash
# BRANCH B -- the gate found degradation. Three measurements, cheapest discriminator first.
#
# The gate can only say that dusk_s1 plays worse than d41_r8. The three ways that can happen
# want opposite fixes, so nothing is decided until they are separated:
#
#   1. the play-time prompt is not the prompt it trained on   (a wiring bug -- check first, and
#      it is the failure this project has actually shipped)
#   2. removing DECK[] cost real information                  (r8 read WITHOUT the segment and
#      never retrained: pure format effect, no training confound)
#   3. the retraining itself was bad                          (r8 and s1 on identical stripped
#      decisions: pure training effect, no format confound)
set -u
REPO=/root/ptcg/repo; cd "$REPO"
export PYTHONPATH=cg-lib HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
SO=$REPO/data/kaggle_engine_ext/libcg_mirror.so
R8=${R8:-/root/out/d41_r8}
OUTDIR=${OUTDIR:-/root/loop_dusk/gate1}
GAMES=${GAMES:-150}
say() { echo "[diag $(date -u +%m-%d_%H:%M:%S)] $*"; }

say "1/3 -- what prompt does the model actually get at play time?"
python3 /root/dusk_diag_prompt.py hf:/root/out/dusk_s1 dusk 2>&1 | tail -12

say "2/3 -- d41_r8 read WITHOUT DECK[], never retrained (pure format effect)"
# Same seeds and same shards as the main gate, so these games pair with the ones already
# played: the arm lands in the same directory and re-pooling picks it up.
SHARD0="marnie_grimmsnarl,alakazam_nz,alakazam"
SHARD1="crustle_geco,crustle,ogerpon_mono"
SHARD2="dudunsparce_box,cynthia_garchomp,dragapult"
SHARD3="mega_lucario_tr,slowking"
i=0
for OPPS in "$SHARD0" "$SHARD1" "$SHARD2" "$SHARD3"; do
  nohup python3 -u tools/gate_protagonist.py \
      --deck dragapult_dusknoir --opp "$OPPS" --games "$GAMES" --seed $((1000 + i * 100)) \
      --arm "engine=engine@prompt" --arm "r8d=hf:$R8@dusk" \
      --out "$OUTDIR/shard${i}b.json" > "$OUTDIR/shard${i}b.log" 2>&1 &
  i=$((i + 1))
done
wait
python3 -u /root/dusk_gate_pool.py "$OUTDIR" 2>&1 | tail -25

say "3/3 -- r8 vs s1 on IDENTICAL decisions (pure training effect)"
python3 -u /root/dusk_diag_rank.py 1500 2>&1 | tail -10

say "diagnosis complete -- read the three sections together, not separately"
