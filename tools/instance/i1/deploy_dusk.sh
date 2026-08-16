#!/usr/bin/env bash
# CPU side: can dusk_s1 actually be submitted?
#
# Speed was the ONLY thing blocking a reranker submission, and the blocker was the prompt: the
# v40 format spent 671 of 838 tokens re-encoding the full deck glossary once per candidate, and
# a cross-encoder has no KV cache to amortise it. v41 plus the DECK-segment removal took the
# state to ~245 tokens, 2.7x shorter, and the projection scales with it. This measures the real
# thing rather than extrapolating -- the deploy numbers on record are gte+v40 and do not carry.
#
# Size moves the other way and needs watching. DeBERTa-v3-base carries a 130,972-row embedding
# (128k sentencepiece + 3,087 domain tokens) against gte's 53,339, so at fp32 the embedding
# alone is ~384 MiB against a 197.66 MiB cap. Weight-only INT8 does not touch it -- it is a
# Gather, not a MatMul -- so the vocab sweep is not an optimisation here, it is the difference
# between shipping and not.
set -u
REPO=/root/ptcg/repo; cd "$REPO"
export PYTHONPATH=cg-lib HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
S1=/root/out/dusk_s1
POOL=$REPO/data/rerank/v41_dusk11.jsonl.gz
W=/root/onnx_dusk
mkdir -p "$W"
say() { echo "[deploy $(date -u +%m-%d_%H:%M:%S)] $*"; }

[ -d "$S1" ] || { say "STOP: no $S1"; exit 1; }
[ -s "$POOL" ] || { say "STOP: no $POOL"; exit 1; }

# ---------------------------------------------------------------- 1. which tokens occur
# Completeness is the whole point: an id outside the kept set maps to [UNK] at inference and
# nothing errors, so a sampled sweep would ship a silent accuracy hole. The corpus is one deck
# now, so the kept set should be SMALLER than the 3,254 the v40 sweep found -- if it comes back
# anywhere near 130k, the sweep read the wrong field and the rest of this is meaningless.
if [ ! -s "$W/keep_ids.json" ]; then
    say "sweeping the token set over the whole dusk pool (3.04M rows)"
    python3 tools/sweep_vocab_rerank.py --data "$POOL" --tokenizer "$S1" \
        --out "$W/keep_ids.json" > "$W/sweep.log" 2>&1 \
        || { say "STOP: sweep failed"; tail -8 "$W/sweep.log"; exit 1; }
fi
KEEP=$(python3 -c "import json;print(len(json.load(open('$W/keep_ids.json'))))")
say "kept ids: $KEEP of 130972 ($(python3 -c "print('%.1f'%(100*$KEEP/130972))")%)"
[ "$KEEP" -lt 60000 ] || { say "STOP: $KEEP kept ids is not a pruned set -- check the sweep"; exit 1; }

# ---------------------------------------------------------------- 2. prune + export + INT8
# One command: prunes the embedding, exports fp32 ONNX, weight-only INT8 blk128 acc_level=1,
# then verifies argmax agreement against the torch model on real decisions and prints the
# compressed budget. max-len 512 because the dusk prompt is ~245 tokens and training capped at
# 512; exporting a 1024 window would cost shape range for nothing.
if [ ! -s "$W/pruned/model_wonly_int8.onnx" ]; then
    say "prune + export + weight-only INT8"
    python3 tools/prune_vocab_rerank.py --model "$S1" --keep "$W/keep_ids.json" \
        --data "$POOL" --work "$W/pruned" --n 60 --max-len 512 > "$W/prune.log" 2>&1 \
        || { say "STOP: prune/export failed"; tail -20 "$W/prune.log"; exit 1; }
fi
grep -aE "^\[[1-4]/4\]|argmax|^BUDGET" "$W/prune.log" | tail -12

# ---------------------------------------------------------------- 3. speed on Kaggle's shape
# 4 threads is the Kaggle CPU; 2 is the insurance reading in case the real machine is slower
# than the spec. The budget is 600 s of inference per game.
ONNX=$W/pruned/model_wonly_int8.onnx
REMAP=$(ls "$W/pruned"/*remap*.npy 2>/dev/null | head -1)
for T in 4 2; do
    say "bench at $T threads"
    python3 tools/bench_rerank_onnx.py --onnx "$ONNX" --tokenizer "$W/pruned/model" \
        --data "$POOL" --n 120 --threads "$T" --max-len 512 \
        ${REMAP:+--remap "$REMAP"} --out "$W/bench_t$T.json" > "$W/bench_t$T.log" 2>&1 \
        || { say "bench at $T threads FAILED"; tail -10 "$W/bench_t$T.log"; continue; }
    grep -aE "mean|p90|max|PROJECTED" "$W/bench_t$T.log" | tail -6
done
say "DEPLOY_DUSK_DONE"
