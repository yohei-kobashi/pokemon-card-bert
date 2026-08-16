#!/bin/bash
# The cross-deck gate for instance1's single-deck reranker.
#
# THREE ARMS, one protagonist, identical shuffles:
#   engine  engine_v2 piloting dragapult_dusknoir            -- the thing to beat
#   r8      d41_r8, the checkpoint before the reformat, read in the OLD prompt format
#   s1      dusk_s1, retrained on the DECK-less single-deck prompt
#
# r8 is in the run because the question is not only "is s1 good" but "did removing DECK[] cost
# anything" -- and that is unanswerable without the same opponents, the same seeds and the same
# seats under the old rendering. Each arm carries its OWN format: reading s1 through the v41
# prompt or r8 through the single-deck one is silent, the win rate merely comes back low.
#
# Sharded by opponent because a 4090 fits several DeBERTa evaluators at once and the games are
# independent; the per-game vectors are written into each shard's json so the paired standard
# error can be rebuilt across shards afterwards.
set -u
REPO=/root/ptcg/repo
cd "$REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

DECK=${DECK:-dragapult_dusknoir}
GAMES=${GAMES:-150}
S1=${S1:-/root/out/dusk_s1}
R8=${R8:-/root/out/d41_r8}
OUTDIR=${OUTDIR:-/root/loop_dusk/gate1}
mkdir -p "$OUTDIR"

# The eleven instance2 trains against. slowking is last and is the one to read with suspicion:
# instance1's pool holds 79,881 rows against it and every one is the PRE-fix decklist.
SHARD0="marnie_grimmsnarl,alakazam_nz,alakazam"
SHARD1="crustle_geco,crustle,ogerpon_mono"
SHARD2="dudunsparce_box,cynthia_garchomp,dragapult"
SHARD3="mega_lucario_tr,slowking"

[ -f "$S1/model.safetensors" ] || { echo "no s1 checkpoint at $S1"; exit 1; }
[ -f "$R8/model.safetensors" ] || { echo "no r8 checkpoint at $R8"; exit 1; }

i=0
for OPPS in "$SHARD0" "$SHARD1" "$SHARD2" "$SHARD3"; do
  nohup python3 -u tools/gate_protagonist.py \
      --deck "$DECK" --opp "$OPPS" --games "$GAMES" --seed $((1000 + i * 100)) \
      --arm "engine=engine@prompt" \
      --arm "r8=hf:$R8@prompt" \
      --arm "s1=hf:$S1@dusk" \
      --out "$OUTDIR/shard$i.json" > "$OUTDIR/shard$i.log" 2>&1 &
  echo "shard $i -> $OPPS (pid $!)"
  i=$((i + 1))
done
wait
echo "=== all shards done ==="
python3 -u /root/dusk_gate_pool.py "$OUTDIR"
