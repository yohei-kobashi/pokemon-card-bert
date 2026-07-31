#!/bin/bash
# Unattended follow-on: wait for a running Stage-A loop, decide, and if the decision is GO
# restart Stage A with LARGER ROUNDS from its best checkpoint.
#
#   bash tools/rl_autochain.sh <workA> <logA> <workC> [mult]
#
# WHY bigger rounds. Two disjoint halves of one round's decisions produce update directions
# only 0.29 apart in cosine: a round's gradient is noise-dominated. The competing explanation
# (the headroom curriculum trading strong cells for weak ones) was tested and does not hold --
# corr(cell delta-vs-engine, its move) = -0.203, t=-0.90, df=19. So more games per round is
# the fix that matches the diagnosis. tools/rl_autostage.py refuses when the diagnosis does
# NOT hold: still improving, worse than the pre-RL start, already at parity, or too few gates.
#
# DISK. A round costs ~354 MB (294 checkpoint + ~60 rollout) and 3x rounds do not change that
# per round, but 12 of them plus Stage A's own 12 will not fit in the headroom. Old rounds are
# pruned as they are superseded, keeping the last KEEP_ROUNDS and never touching a checkpoint
# a gate has blessed.
set -u
WORK_A="${1:?workdir of the running Stage A}"
LOG_A="${2:?its log}"
WORK_C="${3:?workdir for the follow-on}"
MULT="${4:-3}"                       # round-size multiplier
KEEP_ROUNDS="${KEEP_ROUNDS:-3}"
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PWD/cg-lib"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== autochain: waiting for Stage A to finish ($(date -u)) ==="
while pgrep -f "rl_loop.s[h]" > /dev/null; do sleep 120; done
echo "=== Stage A ended $(date -u) ==="

# Never chain off a run that died: the loop prints its own completion line.
if ! grep -q "RL_LOOP_DONE" "$LOG_A" 2>/dev/null; then
  echo "NOGO Stage A did not reach RL_LOOP_DONE (crash / OOM / disk) -- not chaining"
  tail -5 "$LOG_A" 2>/dev/null
  exit 1
fi

DEC=$(python tools/rl_autostage.py "$WORK_A" --stage A 2>/dev/null)
python tools/rl_autostage.py "$WORK_A" --stage A 2>&1 >/dev/null | sed 's/^/  /'   # the series
echo "decision: $DEC"

BEST=$(python tools/rl_autostage.py "$WORK_A" --stage A --print-best 2>/dev/null)

AVAIL=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
if [ "${AVAIL:-0}" -lt 12 ]; then
  echo "STOP only ${AVAIL}G free -- not starting any follow-on"; exit 1
fi

# --- the four outcomes get four DIFFERENT actions (user, 2026-07-28) -----------------
# The diagnosis is only worth making if it changes what happens next; before this the
# script printed four distinct reasons and then did the same nothing four times, idling
# the box overnight in the one case where the right move was unambiguous.
case "$DEC" in
  GO*)
    CKPT=${DEC#GO }
    MODE=bigger
    ;;
  "NOGO still improving"*)
    # Case 1: the current settings are still climbing. Paying 3x per round would be
    # backwards -- keep the size, buy more rounds (12 -> 18). The r15 guard below aborts
    # if it turns and stays turned.
    CKPT="$BEST"; MODE=more_rounds
    ;;
  "NOGO best gate"*)
    # Cases 2 and 3 (below the pre-RL start, or already at parity). Both mean more of the
    # SAME opponent has nothing left to give: engine_v2 is a fixed target, and a policy that
    # has either stopped converging on it or already matched it needs a moving one. Switch to
    # 70% self-play against the frozen best checkpoint (the design's own Stage-A annealing,
    # which we started at 100% heuristic only because a fresh SFT policy needs a fixed
    # competent opponent to climb toward).
    CKPT="$BEST"; MODE=selfplay
    ;;
  *)
    # Case 4: too few gates, or a run whose state we cannot trust.
    echo "STOP $DEC"; exit 0
    ;;
esac
echo "mode: $MODE  from: $CKPT"

# prune Stage A: keep its best checkpoint (we are about to train from it) and the last few
python - "$WORK_A" "$CKPT" "$KEEP_ROUNDS" <<'PY'
import os, re, shutil, sys
work, keep_ckpt, keepn = sys.argv[1], os.path.abspath(sys.argv[2]), int(sys.argv[3])
rounds = sorted(int(m.group(1)) for d in os.listdir(work)
                for m in [re.fullmatch(r"A_r(\d+)_policy", d)] if m)
drop = [r for r in rounds[:-keepn] if os.path.abspath(os.path.join(work, "A_r%d_policy" % r)) != keep_ckpt]
for r in drop:
    shutil.rmtree(os.path.join(work, "A_r%d_policy" % r), ignore_errors=True)
print("pruned %d superseded Stage-A checkpoints (kept %s + last %d)"
      % (len(drop), os.path.basename(keep_ckpt), keepn))
PY

BASE_M=48
case "$MODE" in
  bigger)
    # plateau diagnosed as gradient noise -> more games per round
    mkdir -p "$WORK_C"; cp -r "$WORK_A/baseline" "$WORK_C/baseline" 2>/dev/null || true
    NEW_M=$((BASE_M * MULT))
    echo "=== chaining: bigger rounds, ${NEW_M} matchups (was ${BASE_M}), from $CKPT ==="
    RL_WORK="$WORK_C" RL_MATCHUPS="$NEW_M" RL_ROUNDS_A=12 RL_ROUNDS_B=0 RL_ROUNDS_C=0 \
      RL_EVAL_EVERY=2 RL_EVAL_GAMES=150 RL_BASELINE_DIR="$WORK_C/baseline" \
      bash tools/rl_loop.sh "$CKPT"
    ;;
  more_rounds)
    # SAME work dir: rl_loop skips the rounds already on disk, so this is a true extension
    # of the existing run rather than a restart, and the gate series stays one series.
    echo "=== chaining: same settings, extending 12 -> 18 rounds ==="
    RL_WORK="$WORK_A" RL_MATCHUPS="$BASE_M" RL_ROUNDS_A=18 RL_ROUNDS_B=0 RL_ROUNDS_C=0 \
      RL_EVAL_EVERY=3 RL_EVAL_GAMES=150 RL_BASELINE_DIR="$WORK_A/baseline" \
      RL_ABORT_BELOW_BEST="${RL_ABORT_BELOW_BEST:-0.03}" \
      bash tools/rl_loop.sh "$CKPT"
    ;;
  selfplay)
    # 70% frozen-LM opponents / 30% engine_v2. The opponent is FROZEN at $CKPT for the whole
    # run: chasing the live policy makes the target move under the learner.
    mkdir -p "$WORK_C"; cp -r "$WORK_A/baseline" "$WORK_C/baseline" 2>/dev/null || true
    echo "=== chaining: 70% self-play vs frozen $CKPT ==="
    RL_WORK="$WORK_C" RL_MATCHUPS="$BASE_M" RL_ROUNDS_A=12 RL_ROUNDS_B=0 RL_ROUNDS_C=0 \
      RL_EVAL_EVERY=3 RL_EVAL_GAMES=150 RL_BASELINE_DIR="$WORK_C/baseline" \
      RL_HEURISTIC_FRAC=0.30 RL_OPP_MODEL="$CKPT" \
      bash tools/rl_loop.sh "$CKPT"
    ;;
esac
echo "=== AUTOCHAIN_DONE $(date -u) ==="
