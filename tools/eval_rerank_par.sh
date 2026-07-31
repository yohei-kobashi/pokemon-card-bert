#!/usr/bin/env bash
# Run the fair-protocol reranker eval with one PROCESS PER CELL instead of one process
# looping over all 9 cells.
#
# WHY: eval_rerank.py plays games sequentially and has no resume, so a 9-cell run is a
# ~14-hour all-or-nothing block. ONNX Runtime also scales badly past ~8 intra-op threads on
# a single ~830-token encoder forward, so a single 32-thread process wastes most of a
# 256-core box. 9 cells x 8 threads = 72 cores and ~9x the wall-clock throughput, and each
# cell lands its own JSON so a crash costs one cell, not the run.
#
# TWO MODES. Pass ONNX=torch (or an empty string) to score with PyTorch on CUDA instead:
# that measures MODEL STRENGTH, which is what a v34-vs-v35 comparison needs, and it is far
# faster than the CPU deploy path. Pass a real .onnx to measure the DEPLOY path (quantised,
# vocab-pruned, N threads) -- that is the number the submission decision rests on. They are
# different questions; do not substitute one for the other.
#
# BANK: pass a huge value to measure PURE reranker strength. With the deploy default (480s)
# nearly every game exhausts the bank mid-game and engine_v2 finishes it, so the win rate
# measures the fallback, not the model -- that mistake produced a first, invalid run.
#
# GLOSSARY must match how the training data was rendered (build_rerank --glossary). Getting
# it wrong silently changes the prompt format and measures a model on inputs it never saw.
#
# DECKS/OPPS are overridable from the environment. The submission unit is a DECK, so the
# overall 9-cell number decides nothing on its own -- a model can be far below engine_v2 on
# average and still be the right pilot for one deck. Screening a wider deck set (and then
# re-running the survivors at high N) is the actual decision procedure; the default 3 decks
# are only the historical comparison set. Engine baselines for all 9 submitted decks live in
# /root/out/base_grid300 (tools ref: baseline_grid.py, 300 games/cell).
#
# Usage:
#   tools/eval_rerank_par.sh <outdir> <adapter> <onnx|torch> <remap> [threads] [games] \
#                            [bank_s] [glossary] [deck_mode] [deck_shuffle:0|1]
#   DECKS="ns_zoroark marnie_grimmsnarl" tools/eval_rerank_par.sh ...
set -u
OUT=${1:?outdir}; ADAPTER=${2:?adapter}; ONNX=${3:?onnx or the literal word torch}
REMAP=${4:-}; THREADS=${5:-8}; GAMES=${6:-30}; BANK=${7:-1000000}; GLOSS=${8:-none}; DECKMODE=${9:-static}; DSHUF=${10:-0}
DECKS=${DECKS:-"mega_lucario alakazam_nz_fez crustle_stall"}
OPPS=${OPPS:-"alakazam crustle dragapult"}

mkdir -p "$OUT"
BACKEND_ARG=""
if [ -n "$ONNX" ] && [ "$ONNX" != "torch" ]; then
  BACKEND_ARG="--onnx $ONNX --threads $THREADS"
  [ -n "$REMAP" ] && BACKEND_ARG="$BACKEND_ARG --remap $REMAP"
fi
SHUF_ARG=""
[ "$DSHUF" = "1" ] && SHUF_ARG="--deck-shuffle"

for d in $DECKS; do
  for o in $OPPS; do
    tag="${d}__${o}"
    [ -f "$OUT/$tag.json" ] && { echo "skip $tag (done)"; continue; }
    nohup python3 tools/eval_rerank.py --adapter "$ADAPTER" $BACKEND_ARG \
      --glossary "$GLOSS" --deck-mode "$DECKMODE" $SHUF_ARG --games "$GAMES" --time-budget "$BANK" --decks "$d" --opp "$o" \
      --out "$OUT/$tag.json" > "$OUT/$tag.log" 2>&1 &
    echo "launched $tag pid $!"
  done
done
wait
echo "=== all cells done ==="
python3 - "$OUT" <<'PY'
import json, os, sys
out = sys.argv[1]
w = n = 0
for f in sorted(os.listdir(out)):
    if not f.endswith(".json"):
        continue
    d = json.load(open(os.path.join(out, f)))
    for k, v in d["results"].items():
        print(f"  {k}: {v['win']}/{v['games']} = {v['win_rate']:.1f}%")
        w += v["win"]; n += v["games"]
print(f"OVERALL {w}/{n} = {100.0 * w / max(1, n):.1f}%")
PY
