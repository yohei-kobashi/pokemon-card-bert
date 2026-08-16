#!/bin/bash
# Unattended follow-on to the gate reproducibility measurement.
#
# Waits for the repeat gates on ONE fixed checkpoint to finish, takes their sd, and branches.
# Binomial theory for 21 cells x 150 games is 0.89pt. With 4 points the sd estimate is weak
# (a true 0.89 can read 0.5-2.6), so the cut is set at "clearly abnormal", not at significance.
#
#   sd < 1.5pt    instrument behaves as theory -> RESUME RL (600-game gate, patience rule)
#   sd 1.5-2.0pt  inconclusive -> 4 MORE repeats, re-decide over all of them
#   sd >= 2.0pt   run-level error -> revert the 600-game gate and DIAGNOSE; do not start RL
#
# Scores are recomputed from the repro OUTPUT DIRS rather than scraped from the log: the
# producing script is still running and must not be edited to change its print format.
# Decisions are appended to /root/auto_decision.log.
set -u
cd /root/ptcg/repo
export PYTHONPATH=/root/ptcg/repo:/root/ptcg/repo/cg-lib

DEC=/root/auto_decision.log
BASE=/root/out/rlBIG/baseline
CKPT=/root/out/rlDL/A_r6_policy
say () { echo "[$(date -u +%F' '%H:%M:%S)] $*" | tee -a "$DEC"; }

NCELLS=21
gate_values () {          # one delta per COMPLETE repro dir, in POINTS, on stdout
  local v n out=""
  for d in /root/out/repro_150_*; do
    [ -d "$d" ] || continue
    n=$(ls "$d"/*.json 2>/dev/null | wc -l)
    # rl_gate_score averages whatever cells it finds, so a half-finished directory would be
    # scored over a different (easier or harder) subset and pollute the sd. Require all of them.
    [ "$n" -eq "$NCELLS" ] || continue
    v=$(python tools/rl_gate_score.py "$d" --baseline "$BASE" 2>/dev/null \
          | grep -Eo '^-?[0-9]+\.?[0-9]*$' | tail -1)
    [ -n "$v" ] && out="$out $(python -c "print('%.4f' % ($v*100))")"
  done
  echo "$out"
}

stats () {                # sd mean n  from the values on $1
  python - "$@" <<'PY'
import sys, math
v=[float(x) for x in sys.argv[1:]]
if len(v)<2: print("NA NA %d"%len(v)); raise SystemExit
m=sum(v)/len(v); sd=math.sqrt(sum((x-m)**2 for x in v)/(len(v)-1))
print("%.4f %.4f %d"%(sd,m,len(v)))
PY
}

say "=== auto_after_repro started ==="

# Wait for the producer to finish, but do not wait forever on a dead process.
waited=0
while ! grep -q GATE_REPRO_DONE /root/gate_repro.log 2>/dev/null; do
  if ! pgrep -f "gate_repro.s[h]" >/dev/null; then
    say "gate_repro.sh exited without its DONE marker -- using whatever repeats completed"
    break
  fi
  sleep 60; waited=$((waited+60))
  if [ "$waited" -gt 43200 ]; then say "timed out after 12 h"; exit 1; fi
done

VALS=$(gate_values)
read -r SD MEAN N <<< "$(stats $VALS)"
say "repeats=$N  values(pt):$VALS"
if [ "$SD" = "NA" ]; then say "fewer than 2 usable repeats -- aborting"; exit 1; fi
say "mean ${MEAN}pt  sd ${SD}pt  (binomial prediction 0.89pt)"

if [ "$(python -c "print(1 if 1.5 <= $SD < 2.0 else 0)")" = "1" ]; then
  say "inconclusive band -- running 4 more repeats"
  bash /root/gate_repro.sh "$CKPT" 4 150 >> /root/gate_repro.log 2>&1 || true
  VALS=$(gate_values)
  read -r SD MEAN N <<< "$(stats $VALS)"
  say "after extension: repeats=$N  sd ${SD}pt  values:$VALS"
fi

if [ "$(python -c "print(1 if $SD >= 2.0 else 0)")" = "1" ]; then
  say "DECISION: sd ${SD}pt >= 2.0 -> run-level error. NOT starting RL."
  if sed -i 's|^EVAL_GAMES="${RL_EVAL_GAMES:-600}"|EVAL_GAMES="${RL_EVAL_GAMES:-150}"|' \
        tools/rl_loop.sh; then
    say "reverted rl_loop.sh EVAL_GAMES 600 -> 150 (repeats fix run-level error, bigger runs do not)"
  fi
  say "running the run-variance diagnosis: prompt -> logits -> argmax"
  python /root/diag_run_variance.py "$CKPT" > /root/diag_run_variance.log 2>&1 || true
  grep -E "VERDICT|ARGMAX FLIPS|distinct prompts|bitwise identical|largest logit" \
      /root/diag_run_variance.log 2>/dev/null | sed 's/^/    /' | tee -a "$DEC"
  say "full output: /root/diag_run_variance.log"
  say "STOPPED: fix the instrument before resuming RL."
  exit 0
fi

FREE=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
if [ "${FREE:-0}" -lt 12 ]; then
  say "DECISION: sd ${SD}pt is fine, but only ${FREE}G free -- refusing to start a 12-round run"
  exit 1
fi
say "DECISION: sd ${SD}pt < 1.5 -> consistent with theory. RESUMING RL."
say "  from $CKPT, 72 matchups, branching on, 600 games/cell every 4 rounds, patience rule"
say "  ~44 min/round + ~1.9 h/gate => roughly 14 h for 12 rounds unless it plateaus first"
RL_WORK=/root/out/rlDL2 RL_MATCHUPS=72 RL_ROUNDS_A=12 RL_ROUNDS_B=0 RL_ROUNDS_C=0 \
RL_EVAL_EVERY=4 RL_EVAL_GAMES=600 RL_BASELINE_DIR="$BASE" \
RL_BRANCH=8 RL_BRANCH_K=4 RL_BRANCH_PLAYOUTS=2 RL_BRANCH_WEIGHT=1.0 RL_DECISION_FRAC=0.5 \
  bash tools/rl_loop.sh "$CKPT" > /root/rlDL2.log 2>&1 || say "rl_loop exited non-zero"
say "RL finished; gates:"
sed 's/^/    /' /root/out/rlDL2/A_gates.txt 2>/dev/null | tee -a "$DEC"
say "=== auto_after_repro done ==="
