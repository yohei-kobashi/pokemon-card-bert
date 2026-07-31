#!/usr/bin/env bash
# Turn a trained reranker into a measured, submittable bundle. One command, because the
# steps are order-dependent and each one silently poisons the next if skipped:
#
#   sweep  -> which token ids can ever appear (FORMAT-DEPENDENT: re-run after any prompt change)
#   prune  -> slice the embedding to those ids, export fp32 ONNX, weight-only-INT8 quantize,
#             and verify against the full-vocab PyTorch reference
#   bench  -> per-decision latency at 4 threads = the competition's vCPU count
#   bundle -> tar.gz, checked against the 197.65625 MiB cap (COMPRESSED)
#
# Win rate is NOT run here: it takes hours and wants tools/eval_rerank_par.sh across 9 cells.
#
# DECK_MODE/DECK_SHUFFLE must match the flags build_rerank.py rendered DATA with, or main.py
# builds a prompt the model never saw -- silently, with no error and no size change.
#
# Usage:
#   tools/rerank_deploy.sh <model_dir> <data.jsonl.gz> <work_dir> <deck> <tag> [glossary] \
#                          [deck_mode] [deck_shuffle:0|1]
set -euo pipefail
MODEL=${1:?trained reranker dir}
DATA=${2:?rerank jsonl.gz the model was trained on}
WORK=${3:?scratch dir}
DECK=${4:?deck name}
TAG=${5:?submission tag}
GLOSSARY=${6:-none}
DECK_MODE=${7:-static}
DECK_SHUFFLE=${8:-0}

cd "$(dirname "$0")/.."
export PYTHONPATH=cg-lib

echo "=== [1/4] vocab sweep ==="
python3 tools/sweep_vocab_rerank.py --data "$DATA" --tokenizer "$MODEL" \
  --out "$WORK/keep_ids.json"

echo; echo "=== [2/4] prune + export + weight-only INT8 ==="
python3 tools/prune_vocab_rerank.py --model "$MODEL" --keep "$WORK/keep_ids.json" \
  --data "$DATA" --work "$WORK/pruned" --n 40

echo; echo "=== [3/4] CPU speed at 4 threads (Kaggle runtime) ==="
python3 tools/bench_rerank_onnx.py --onnx "$WORK/pruned/model_wonly_int8.onnx" \
  --tokenizer "$MODEL" --remap "$WORK/pruned/model/vocab_remap.npy" \
  --data "$DATA" --n 60 --threads 4 --out "$WORK/bench_t4.json"

echo; echo "=== [4/4] submission bundle ==="
SHUF_ARG=""
[ "$DECK_SHUFFLE" = "1" ] && SHUF_ARG="--deck-shuffle"
python3 tools/build_rerank_submission.py "$DECK" \
  --onnx "$WORK/pruned/model_wonly_int8.onnx" --tokenizer "$MODEL" \
  --remap "$WORK/pruned/model/vocab_remap.npy" --glossary "$GLOSSARY" \
  --deck-mode "$DECK_MODE" $SHUF_ARG --tag "$TAG"

echo
echo "NEXT: win rate over the 9 fair-protocol cells (hours, run it in parallel):"
echo "  tools/eval_rerank_par.sh <outdir> $MODEL \\"
echo "    $WORK/pruned/model_wonly_int8.onnx $WORK/pruned/model/vocab_remap.npy 8 30 1000000"
echo "  (pass a HUGE bank: at the deploy default of 480s most games exhaust it mid-game and"
echo "   engine_v2 finishes them, so the number measures the fallback, not the model)"
