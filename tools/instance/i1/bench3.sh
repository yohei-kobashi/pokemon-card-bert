#!/usr/bin/env bash
# Dynamic INT8 destroyed DeBERTa (argmax 10.0%, BELOW the ~15% chance line at 6.6 candidates).
# That is the failure quant_weightonly_rerank.py was written for: dynamic INT8 quantizes
# ACTIVATIONS too, and DeBERTa-v3's disentangled attention carries larger activation outliers
# than ModernBERT did. Redo it on the validated path: sweep -> prune embedding -> weight-only
# INT8 (block 128, accuracy_level=1, activations stay fp32), which held 97.5% argmax on gte.
set -u
say() { echo "[b3 $(date -u +%m-%d_%H:%M:%S)] $*"; }
cd /root/ptcg/repo
DATA=/root/ptcg/repo/data/rerank/deberta41_r7.jsonl.gz
CKPT=/root/out/d41_r6

say "vocab sweep (v41 format -- the format decides the token set)"
PYTHONPATH=cg-lib python3 tools/sweep_vocab_rerank.py --data "$DATA" \
  --tokenizer "$CKPT" --out /root/onnx/keep_ids_d41.json > /root/b3_sweep.log 2>&1 \
  || { say "SWEEP FAILED"; tail -20 /root/b3_sweep.log; exit 1; }
tail -6 /root/b3_sweep.log

say "prune + export + weight-only INT8"
PYTHONPATH=cg-lib python3 tools/prune_vocab_rerank.py --model "$CKPT" \
  --keep /root/onnx/keep_ids_d41.json --data "$DATA" --work /root/onnx/d41_wonly \
  --n 60 --max-len 512 > /root/b3_prune.log 2>&1 \
  || { say "PRUNE FAILED"; tail -30 /root/b3_prune.log; exit 1; }
tail -20 /root/b3_prune.log
ls -la /root/onnx/d41_wonly/

M=$(ls /root/onnx/d41_wonly/*.onnx 2>/dev/null | head -1)
say "bench $M"
for T in 2 4; do
  python3 tools/bench_rerank_onnx.py --onnx "$M" --tokenizer "$CKPT" \
    --data "$DATA" --n 60 --threads $T 2>&1 | tail -10
done
say BENCH3_DONE
