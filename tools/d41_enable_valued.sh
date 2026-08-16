#!/usr/bin/env bash
# Fold the regenerated v41 attach-valued records into the DeBERTa loop, at the round boundary.
#
# WHY A RESTART IS NEEDED. dagger_loop8.sh reads VALUED / VALUED_FRAC once, as shell variables
# at launch. The loop was started before tools/attach_label.py had produced a v41 file, so it
# is running with VALUED_FRAC=0 -- deliberately, because the only valued files that existed
# were v40-rendered and would have put 5% of every round in the wrong prompt format. The file
# now exists (16,830 records, 65 decks, 96.0% carrying a v41 fact), so the loop should pick it
# up. A running bash cannot be told; it has to be relaunched.
#
# WHY AT THE BOUNDARY. Round 1 is ~5 hours of training. Killing mid-round throws that away.
# This waits for the loop to WRITE round 1's checkpoint and print its completion line, then
# relaunches at round 2 from that checkpoint. Nothing is lost: dagger_loop8 reuses an existing
# mirror_r<N>.json, so even a screen that had started is not repeated.
#
# The wait condition is BOTH the log line and the safetensors file. The log line alone would
# fire on a round that printed and then failed to save; the file alone has no round number.
#
#   nohup bash tools/d41_enable_valued.sh > /root/d41_valued.log 2>&1 &
set -u
REPO=${REPO:-/root/ptcg/repo}
STATE=${STATE:-/root/loop_deberta41}
ROUND=${ROUND:-1}                       # the round to wait for
NEXT=$((ROUND + 1))
CKPT=${CKPT:-/root/out/d41_r$ROUND}
VALUED=${VALUED:-$REPO/data/rerank/v41_attach.jsonl.gz}
VALUED_FRAC=${VALUED_FRAC:-0.05}
RUNCOPY=${RUNCOPY:-/root/d41_run.sh}
say() { echo "[valued $(date -u +%m-%d_%H:%M:%S)] $*"; }

[ -s "$VALUED" ] || { say "no valued file at $VALUED -- nothing to enable"; exit 1; }
say "waiting for round $ROUND to finish (log line + $CKPT/model.safetensors)"

while true; do
  if grep -q "round $ROUND done -> $CKPT" "$STATE/loop.log" 2>/dev/null \
     && [ -f "$CKPT/model.safetensors" ]; then
    say "round $ROUND is complete"
    break
  fi
  if ! pgrep -f "$RUNCOPY" > /dev/null 2>&1; then
    say "the loop is no longer running and round $ROUND never completed -- STOPPING so a"
    say "half-finished state is not papered over. Inspect $STATE/loop.log."
    exit 1
  fi
  sleep 120
done

# Kill the loop only AFTER its round-1 checkpoint is on disk. pkill -f the frozen copy path so
# this cannot match the editor, this script, or an unrelated python.
say "stopping the loop to relaunch with VALUED_FRAC=$VALUED_FRAC"
pkill -f "$RUNCOPY" || true
sleep 5
pkill -f "tools/train_rerank.py --data .*deberta41" 2>/dev/null || true
sleep 5

cd "$REPO"
cp tools/dagger_loop8.sh "$RUNCOPY"     # refresh the frozen copy; scp over a running bash kills it
KIND=deberta41 MODEL="$CKPT" OUTSTEM=/root/out/d41_r START_ROUND=$NEXT DEADLINE_H=72 \
  VALUED="$VALUED" VALUED_FRAC="$VALUED_FRAC" \
  setsid nohup bash "$RUNCOPY" >> /root/d41_start.log 2>&1 < /dev/null &
sleep 20

if pgrep -f "$RUNCOPY" > /dev/null 2>&1; then
  say "relaunched at round $NEXT from $CKPT with valued $(basename "$VALUED") @ $VALUED_FRAC"
  tail -3 "$STATE/loop.log"
else
  say "RELAUNCH FAILED -- no loop process. $STATE/loop.log and /root/d41_start.log have the reason."
  exit 1
fi
