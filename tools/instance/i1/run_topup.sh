#!/bin/bash
cd ~/ptcg/repo
chmod 755 cg-lib/cg/*.so 2>/dev/null
export PYTHONPATH="$PWD:$PWD/cg-lib"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo '=== SFT honch_aug MIXED top-up start ==='; date
time python tools/sft_train_eval.py \
  --data data/sft/topup_mixed.jsonl.gz \
  --adapter out/lora_v34 --resume \
  --max-samples 200000 --deadline-h 5 \
  --batch 4 --accum 4 --lr 5e-5 --lora-r 16 \
  --eval-decks rockets_honchkrow,mega_lucario,crustle_stall \
  --eval-opp alakazam,crustle,dragapult --eval-games 30 2>&1
echo '=== topup done ==='; date
