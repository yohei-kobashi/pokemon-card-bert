#!/usr/bin/env bash
# Does the DeBERTa reranker fit the REAL Kaggle box? The resource page says 2 vCPUs, not the
# 4 the earlier bench assumed, and the earlier 984 s/game figure was v40 prompts + gte -- a
# different model AND a ~2.4x longer prompt, so it does not transfer either way. Measure.
set -u
say() { echo "[b2 $(date -u +%m-%d_%H:%M:%S)] $*"; }
cd /root/ptcg/repo
DATA=/root/ptcg/repo/data/rerank/deberta41_r7.jsonl.gz
CKPT=/root/out/d41_r6

say "token stats on the v41 records the model actually sees"
python3 - "$DATA" "$CKPT" <<'PY'
import gzip, json, sys, statistics as st
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[2])
lens, cands = [], []
for i, line in enumerate(gzip.open(sys.argv[1], "rt")):
    if i % 977: continue
    d = json.loads(line)
    st_txt = d.get("state") or d.get("prompt") or ""
    cs = d.get("cands") or d.get("candidates") or []
    cands.append(len(cs))
    if cs:
        lens.append(len(tok(st_txt, cs[0])["input_ids"]))
    if len(lens) >= 200: break
lens.sort()
print("pair tokens: mean %.0f p50 %d p90 %d p99 %d max %d | candidates mean %.2f p90 %d max %d"
      % (st.mean(lens), lens[len(lens)//2], lens[int(.9*len(lens))], lens[int(.99*len(lens))],
         lens[-1], st.mean(cands), sorted(cands)[int(.9*len(cands))], max(cands)))
PY

say "export d41_r6 -> ONNX INT8"
python3 tools/export_rerank_onnx.py --model "$CKPT" --out /root/onnx/d41 \
  --data "$DATA" --verify-n 60 > /root/bench2_export.log 2>&1 \
  || { say "EXPORT FAILED"; tail -25 /root/bench2_export.log; exit 1; }
tail -12 /root/bench2_export.log
ls -la /root/onnx/d41/

for T in 2 4; do
  say "bench threads=$T"
  python3 tools/bench_rerank_onnx.py --onnx /root/onnx/d41/model_int8.onnx \
    --tokenizer "$CKPT" --data "$DATA" --n 60 --threads $T 2>&1 | tail -14
done
say BENCH2_DONE
