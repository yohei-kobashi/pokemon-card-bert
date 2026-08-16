#!/bin/bash
# How reproducible is the RL gate? Score ONE fixed checkpoint N times and take the spread.
#
# Needed because the same checkpoint scored -4.37 and -6.97 on two runs (2.60pt apart), which
# retired the "gates cluster tightly so they are precise" argument. Binomial theory says
# 21 cells x 150 games -> SE 0.89pt per gate. If the measured sd matches, 1/sqrt(N) scaling is
# trustworthy and the 600-game size can be chosen analytically instead of measured (which
# would cost ~6 h). If the measured sd is much larger, there is a non-sampling error source
# and more games will NOT fix it.
set -e
cd /root/ptcg/repo
export PYTHONPATH=/root/ptcg/repo:/root/ptcg/repo/cg-lib

CKPT="${1:-/root/out/rlDL/A_r6_policy}"
REPS="${2:-4}"
GAMES="${3:-150}"
BASE="${RL_BASELINE_DIR:-/root/out/rlBIG/baseline}"

pf () { python -c "import sys;sys.path.insert(0,'tools');import rl_config;print($1)"; }
GLOSS=$(pf "rl_config.PROMPT_FMT['glossary']")
DMODE=$(pf "rl_config.PROMPT_FMT['deck_mode']")
DSHUF=$(pf "int(rl_config.PROMPT_FMT['deck_shuffle'])")
DECKS_S=$(pf "' '.join(rl_config.STAGE_C_TARGETS)")

echo "checkpoint : $CKPT"
echo "repeats    : $REPS at $GAMES games/cell"
echo "baseline   : $BASE"
echo

for i in $(seq 1 "$REPS"); do
  OUT="/root/out/repro_${GAMES}_${i}"
  rm -rf "$OUT"
  DECKS="$DECKS_S" OPPS="alakazam crustle dragapult" \
    tools/eval_rerank_par.sh "$OUT" "$CKPT" torch "" 8 "$GAMES" 1000000 \
      "$GLOSS" "$DMODE" "$DSHUF" > "$OUT.log" 2>&1 || true
  D=$(python tools/rl_gate_score.py "$OUT" --baseline "$BASE" 2>>"$OUT.log")
  echo "REPRO $GAMES rep $i : $D"
done
echo "GATE_REPRO_DONE"
