cd ~/ptcg/repo
chmod 755 cg-lib/cg/*.so 2>/dev/null
export PYTHONPATH="$PWD:$PWD/cg-lib"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== assemble v2 corpus (gzip multi-stream concat) ==="
cat data/sft_v2/v34_full.jsonl.gz data/sft_v2/meta_topup_0723.jsonl.gz data/sft_v2/mega_starmie_v2.jsonl.gz > data/sft_v2/all_v2.jsonl.gz
ls -la data/sft_v2/all_v2.jsonl.gz
echo "=== warm-start from v34 (keep weights+embeddings, fresh optimizer/step) ==="
rm -rf out/lora_v35; cp -r out/lora_v34 out/lora_v35
rm -f out/lora_v35/opt_state.pt out/lora_v35/sft_progress.json
echo "=== re-SFT v2 (balanced, fla) start ==="; date
time python tools/sft_train_eval.py \
  --data data/sft_v2/all_v2.jsonl.gz \
  --adapter out/lora_v35 --resume \
  --balance-decks --per-deck 4000 --deadline-h 10 \
  --batch 4 --accum 4 --lr 1e-4 --lora-r 16 \
  --eval-decks mega_starmie,alakazam_nz,crustle_stall \
  --eval-opp alakazam_nz,archaludon,marnie_grimmsnarl --eval-games 30 2>&1
echo "=== re-SFT v2 done ==="; date
