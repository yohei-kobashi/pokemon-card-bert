#!/bin/bash
# Assemble the Kaggle submission bundle on the vast box.
set -e
REPO=/root/ptcg/repo
OUT=/root/subm_crustle
rm -rf "$OUT"; mkdir -p "$OUT"

# python packages the agent imports at runtime
cp -r "$REPO/lm" "$OUT/lm"
cp -r "$REPO/cg-lib/cg" "$OUT/cg"
mkdir -p "$OUT/agents"
cp "$REPO/agents/__init__.py" "$REPO/agents/engine_v2.py" "$REPO/agents/_engine.py" \
   "$REPO/agents/tuning.json" "$OUT/agents/"
mkdir -p "$OUT/decks"
cp "$REPO/decks/crustle.csv" "$OUT/decks/"

# bundled llama-cpp-python (glibc 2.35 / py3.11, matches Kaggle)
cp -r /opt/conda/lib/python3.11/site-packages/llama_cpp "$OUT/llama_cpp"

# model + entrypoint
cp /root/sftv2.Q4_K_M.gguf "$OUT/model.gguf"
cp /root/main_lm.py "$OUT/main.py"

# drop __pycache__ / tests to keep it lean
find "$OUT" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
rm -f "$OUT/lm/roundtrip_test.py" 2>/dev/null || true

echo "=== bundle tree (top) ==="
ls -la "$OUT"
echo "=== sizes ==="
du -sh "$OUT" "$OUT"/model.gguf "$OUT"/llama_cpp
echo STAGED
