#!/usr/bin/env bash
# Two bundles of the same model: pure LM, and LM with attach handed to engine_v2.
#
# Why the second one exists. The model's attach decisions rank at 16-29% top1 against a 14%
# chance baseline -- it is not choosing, it is guessing -- and routing ONLY attach to the
# heuristic was worth +11.4pt when it was measured. That measurement lived in the experimental
# harness; lm/agent.py, the module that actually ships, had no way to express it until today.
# Building both means the choice can be made on games rather than on the remembered number.
set -u
REPO=/root/ptcg/repo; cd "$REPO"
export PYTHONPATH=cg-lib HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
W=/root/onnx_dusk
say() { echo "[bundle $(date -u +%m-%d_%H:%M:%S)] $*"; }

for VARIANT in pure defer; do
    if [ "$VARIANT" = pure ]; then DEFER=""; TAG=dusk_s1_pure; else DEFER=attach; TAG=dusk_s1_attach; fi
    say "=== building $TAG (defer='${DEFER:-none}') ==="
    python3 tools/build_rerank_submission.py dragapult_dusknoir \
        --onnx "$W/pruned/model_wonly_int8.onnx" \
        --tokenizer "$W/pruned/model" \
        --remap "$W/pruned/model/vocab_remap.npy" \
        --pfmt dusk --defer "$DEFER" --tag "$TAG" \
        --threads 4 --max-len 512 --time-budget 480 \
        --out /root/subm > "/root/subm_$TAG.log" 2>&1 \
        || { say "$TAG BUILD FAILED"; tail -25 "/root/subm_$TAG.log"; continue; }
    grep -aE "prompt format|deferred to|selfcheck|SELFCHECK|MiB|cap|TOTAL" "/root/subm_$TAG.log" | tail -14
    ls -la "/root/subm/$TAG.tar.gz"
done
say "BUILD_BUNDLES_DONE"
