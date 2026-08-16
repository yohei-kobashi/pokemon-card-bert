#!/usr/bin/env bash
# Speed of the pruned + weight-only-INT8 dusk_s1 on a Kaggle-shaped CPU.
#
# 4 threads is the Kaggle machine; 2 is the insurance reading. The budget is 600 s of inference
# per game, and a game is ~80 scored decisions (the 65 figure that produced the old 433 s
# projection came from dividing training records by games, which undercounts by ~25% because
# build_rerank keeps winner-side records only and drops decisions whose candidates dedupe below
# two).
set -u
REPO=/root/ptcg/repo; cd "$REPO"
export PYTHONPATH=cg-lib HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
W=/root/onnx_dusk
POOL=$REPO/data/rerank/v41_dusk11.jsonl.gz
REMAP=$W/pruned/model/vocab_remap.npy
[ -s "$REMAP" ] || { echo "no remap at $REMAP"; exit 1; }
for T in 4 2; do
    echo "=== threads $T ==="
    python3 tools/bench_rerank_onnx.py --onnx "$W/pruned/model_wonly_int8.onnx" \
        --tokenizer "$W/pruned/model" --data "$POOL" --n 120 --threads "$T" \
        --max-len 512 --remap "$REMAP" --out "$W/bench_t$T.json" 2>&1 | tail -25
done
echo "BENCH_DUSK_DONE"
