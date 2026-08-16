#!/usr/bin/env bash
# v35 -> submittable bundle. Same four steps as tools/rerank_deploy.sh, run apart because the
# token sweep must see BOTH files v35 trained on while prune/bench take a single path (bench
# opens it with gzip.open, so a comma list would break there).
#
# v35's prompt format is static / NO shuffle. Fingerprinted from the data itself, not guessed:
# curengine_0724_v2 carries all 60 cards in DECK[] in non-ascending (decklist-file) order,
# while _rem is ascending and _v36 is per-decision shuffled. Passing --deck-mode remaining
# here would feed the model a prompt shape it never trained on, silently.
set -euo pipefail
cd /root/ptcg/repo
export PYTHONPATH=cg-lib
MODEL=/root/out/rerank_gte_v35
WORK=/root/onnx_v35
D1=/root/data/rerank/curengine_0724_v2.rerank.jsonl.gz
D2=/root/data/rerank/v34_full_v2.rerank.jsonl.gz
mkdir -p "$WORK"

echo "=== [1/4] vocab sweep (both v35 data files) ==="
python3 tools/sweep_vocab_rerank.py --data "$D1,$D2" --tokenizer "$MODEL" \
  --out "$WORK/keep_ids.json"

echo
echo "=== [2/4] prune + export + weight-only INT8 ==="
python3 tools/prune_vocab_rerank.py --model "$MODEL" --keep "$WORK/keep_ids.json" \
  --data "$D1" --work "$WORK/pruned" --n 40

echo
echo "=== [3/4] CPU speed at 4 threads (Kaggle runtime) ==="
python3 tools/bench_rerank_onnx.py --onnx "$WORK/pruned/model_wonly_int8.onnx" \
  --tokenizer "$MODEL" --remap "$WORK/pruned/model/vocab_remap.npy" \
  --data "$D1" --n 60 --threads 4 --out "$WORK/bench_t4.json"

echo
echo "=== [4/4] submission bundle ==="
python3 tools/build_rerank_submission.py crustle_stall \
  --onnx "$WORK/pruned/model_wonly_int8.onnx" --tokenizer "$MODEL" \
  --remap "$WORK/pruned/model/vocab_remap.npy" --glossary none --deck-mode static \
  --tag rr_v35_crustle
echo "=== DEPLOY_V35_DONE ==="
