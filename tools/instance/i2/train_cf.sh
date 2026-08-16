#!/bin/bash
# Qwen3-4B, card-first answers, v39 prompts + 5% DAgger.
#
# Settings that were MEASURED rather than inherited:
#   bsz 8 / accum 4     batch size makes no difference to throughput (8/16/32 within 1%), so it
#                       is chosen for the optimisation, matching the 9B recipe's effective 32.
#   grad checkpointing  left ON: a clean idle-box A/B put ON and OFF within 1% of each other
#                       (129.2/129.1 vs 129.1/130.3 s), so the memory it saves is free.
#   --group-by-length   removes 22.5% of padded tokens (470 -> 364 per sample, counted in the
#                       real dataloader). It buys little wall clock here, but it costs nothing.
#   maxlen 896          longest prompt in the mix is 836; nothing truncates.
#   emb-lr-mult 1       the added rows already moved 30.2% of their norm in 40 steps on this
#                       model, so the reranker's stuck-embedding problem is not present and a
#                       multiplier would only risk destabilising them.
#   limit 400000        ~15 h at the measured 7.43 samples/s. Checkpoints every 1000 steps mean
#                       an earlier one can be evaluated without waiting for the end.
set -u
cd /root/ptcg/repo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 tools/instance/sft_teacher.py \
  --model unsloth/Qwen3-4B-Base \
  --data data/sft/v39_dag005.jsonl.gz \
  --domain-tokens --card-first data/cardfirst_v39.json \
  --out /root/out/qwen3_4b_cf1 \
  --limit 400000 --eval-n 4000 --epochs 1 \
  --bsz 8 --accum 4 --maxlen 896 --group-by-length \
  --save-steps 1000
echo "TRAIN DONE rc=$?"
