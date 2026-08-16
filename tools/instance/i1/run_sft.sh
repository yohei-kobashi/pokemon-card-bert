#!/bin/bash
cd ~/ptcg/repo
chmod 755 cg-lib/cg/*.so 2>/dev/null
export PYTHONPATH="$PWD:$PWD/cg-lib"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== SFT v34 BALANCED + fla fast-path (resume) start ==="; date
time python tools/sft_train_eval.py \
  --data data/sft/v34_full.jsonl.gz \
  --adapter out/lora_v34 \
  --resume --balance-decks --per-deck 4000 \
  --deadline-h 10 \
  --batch 4 --accum 4 --lr 1e-4 --lora-r 16 \
  --eval-decks mega_lucario,alakazam_nz_fez,crustle_stall \
  --eval-opp alakazam,crustle,dragapult \
  --eval-games 30 2>&1
echo "=== SFT v34 done ==="; date
