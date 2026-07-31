#!/bin/bash
# vast.ai: environment for the CROSS-ENCODER agent, then a smoke train + a win rate vs
# engine_v2 so a fresh box proves itself end to end.
#
# Assumes ~/ptcg/repo (code) and the rerank data under /root/data/rerank/ are already on the
# box. (~/.kaggle on vast holds only access_token, so the Kaggle CLI cannot pull them.)
#
# Usage:  bash tools/vast_setup.sh [rerank_data.jsonl.gz] [base_or_ckpt]
#
# REWRITTEN 2026-07-28. The previous version installed the Qwen3.5 decoder stack -- Gated
# DeltaNet needs flash-linear-attention + causal-conv1d + a Triton newer than the one torch
# pins, an install order its own comment called "fragile" because it was. None of it applies
# to a ModernBERT encoder: torch + transformers is the whole dependency set. The decoder path
# is quarantined in tools/_legacy_decoder/ (dead on size AND on strength; see its README).
set -e

DATA="${1:-/root/data/rerank/curengine_0724_v36.rerank.jsonl.gz}"
MODEL="${2:-Alibaba-NLP/gte-reranker-modernbert-base}"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# onnxruntime is for the DEPLOY path (lm/rerank_scorer.py + the submission bundle), not for
# training; installed here so one box can do both.
pip install -q -U "transformers>=5.12.1" accelerate onnxruntime onnx tokenizers safetensors

cd ~/ptcg/repo
chmod 755 cg-lib/cg/*.so
export PYTHONPATH="$PWD:$PWD/cg-lib"

echo "=== gating test: serializer / action round-trip / agent, incl. the SHIPPED prompt format ==="
python lm/roundtrip_test.py

echo "=== smoke train (tiny) ==="
python tools/train_rerank.py --data "$DATA" --model "$MODEL" --out out/rerank_smoke \
  --cap-matchup 8 --max-samples 4000 --max-len 640 --pair-batch 128 --accum 2 \
  --grad-ckpt --deadline-h 0.2

echo "=== win rate vs engine_v2 (protocol decks, few games -- a smoke, not a measurement) ==="
python tools/eval_rerank.py --adapter out/rerank_smoke --games 6 \
  --decks mega_lucario,alakazam_nz_fez,crustle_stall --opp alakazam,crustle,dragapult \
  --glossary none --deck-mode remaining --deck-shuffle
