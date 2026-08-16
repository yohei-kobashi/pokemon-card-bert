#!/usr/bin/env bash
set -u
cd /root/ptcg/repo
for T in 2 4; do
  python3 tools/bench_rerank_onnx.py --onnx /root/onnx/d41_wonly/model_wonly_int8.onnx \
    --tokenizer /root/onnx/d41_wonly/model --remap /root/onnx/d41_wonly/model/vocab_remap.npy \
    --data /root/ptcg/repo/data/rerank/deberta41_r7.jsonl.gz --n 60 --threads $T --max-len 512 2>&1 | tail -9
done
echo BENCH4_DONE
