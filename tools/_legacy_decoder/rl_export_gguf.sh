#!/bin/bash
# Export the final RL policy for FAST CPU inference on Kaggle (memory lm-agent-plan-status:
# GGUF via llama.cpp supports qwen3_5; thread=physical-cores + KV-reuse = 5ms/move).
#
#   bash rl_export_gguf.sh <BASE_MODEL> <FINAL_ADAPTER_DIR> [OUT_DIR]
#
# Steps (docs decision — ship Q6_K/Q8_0, NOT Q4: the model is tiny, Kaggle RAM ample,
# clock generous, so quality>size; imatrix built on REAL game states preserves the
# tensors that matter for THIS task):
#   1. merge LoRA -> fp16 HF model
#   2. HF -> GGUF f16
#   3. build an importance matrix from real serialized game-state prompts
#   4. quantize -> Q6_K and Q8_0 (pick the higher-winrate one; both fit ~1GB RAM)
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PWD/cg-lib"
BASE="${1:?usage: rl_export_gguf.sh <BASE> <ADAPTER> [OUT]}"
ADAPTER="${2:?need adapter}"
OUT="${3:-out/gguf}"; mkdir -p "$OUT"
LCPP="${LLAMA_CPP:-$HOME/llama.cpp}"     # clone of ggml-org/llama.cpp (has convert + quantize)

echo "=== [1/4] merge LoRA -> fp16 ==="
python tools/merge_adapter.py --base "$BASE" --adapter "$ADAPTER" --out "$OUT/merged_fp16"

echo "=== [2/4] HF -> GGUF f16 ==="
python "$LCPP/convert_hf_to_gguf.py" "$OUT/merged_fp16" --outfile "$OUT/model-f16.gguf" --outtype f16

echo "=== [3/4] importance matrix from REAL game-state prompts ==="
# ~2k prompts sampled from the SFT/rollout data -> calibrates the quantizer for our task.
python - "$OUT/imatrix_calib.txt" <<'PY'
import glob, gzip, json, os, random, sys
out = sys.argv[1]; rng = random.Random(0); n = 0
srcs = sorted(glob.glob("data/sft/*v34*.jsonl.gz")) or sorted(glob.glob("out/rl/*.jsonl.gz"))
with open(out, "w") as w:
    for s in srcs:
        op = gzip.open(s, "rt")
        for line in op:
            d = json.loads(line)
            p = d.get("prompt") or ""
            if p and rng.random() < 0.02:
                w.write(p.replace("\n", " ") + "\n"); n += 1
            if n >= 2000:
                break
        if n >= 2000:
            break
print("calib prompts:", n)
PY
"$LCPP/llama-imatrix" -m "$OUT/model-f16.gguf" -f "$OUT/imatrix_calib.txt" \
    -o "$OUT/imatrix.dat" --chunks 200 2>&1 | tail -3 || echo "imatrix skipped (llama-imatrix missing)"

echo "=== [4/4] quantize -> Q4_K_M + Q6_K + Q8_0 (imatrix-guided if available) ==="
# Q4_K_M added (2026-07-22): CPU inference is memory-bandwidth bound, so lower bit-width is
# FASTER on CPU (Q4 ~0.5GB vs Q8 ~0.85GB for 0.8B ~= 40% less bytes/token). Emit all three
# and pick at ship time by measuring CPU speed x winrate on a Kaggle-equivalent CPU -- the
# RL training precision (now bf16) does NOT affect this; the ship-quant is the CPU-speed lever.
IM=""; [ -f "$OUT/imatrix.dat" ] && IM="--imatrix $OUT/imatrix.dat"
"$LCPP/llama-quantize" $IM "$OUT/model-f16.gguf" "$OUT/model-Q4_K_M.gguf" Q4_K_M
"$LCPP/llama-quantize" $IM "$OUT/model-f16.gguf" "$OUT/model-Q6_K.gguf" Q6_K
"$LCPP/llama-quantize" $IM "$OUT/model-f16.gguf" "$OUT/model-Q8_0.gguf" Q8_0
ls -la "$OUT"/*.gguf
echo "DONE. Ship-quant decision = CPU-SPEED x WINRATE bench on a Kaggle-equivalent CPU:"
echo "  for each quant, load in llama-cpp-python (threads=physical cores, KV-reuse across"
echo "  candidates+turns) -> measure ms/move AND winrate vs engine_v2; ship the best speed/"
echo "  quality point. If Q4 wins on speed but loses quality, THEN consider GGUF-matched"
echo "  quant-aware training (NOT bnb-NF4). See lm-agent-plan-status / rl-pipeline-scaffold."
