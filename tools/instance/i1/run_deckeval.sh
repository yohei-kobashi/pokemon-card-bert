cd ~/ptcg/repo
export PYTHONPATH="$PWD:$PWD/cg-lib" TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== deck-selection eval (lora_v35 v2 vs current meta) ==="; date
python tools/sft_train_eval.py --skip-train --adapter out/lora_v35 \
  --eval-decks dragapult,dragapult_dusknoir,rockets_mewtwo,crustle,marnie_grimmsnarl,alakazam \
  --eval-opp alakazam_nz,marnie_grimmsnarl,archaludon --eval-games 20 2>&1
echo DECKEVAL_DONE; date
