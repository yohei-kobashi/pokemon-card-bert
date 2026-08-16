#!/bin/bash
# Swap instance2 onto the tuned loop at the round-2 screen boundary.
#
# Waiting for the screen rather than acting now: round 2's screen is already 3.8 hours in flight
# and re-running it would cost more than the change saves. The screen's output is reused, so the
# handoff is free -- the only thing that changes is the batch geometry of the training that
# follows.
#
# Not an in-place edit: bash reads a running script incrementally by byte offset.
set -u
LOG=/root/handoff_i2b.log
exec >> "$LOG" 2>&1
say() { echo "[hi2b $(date -u +%m-%d_%H:%M:%S)] $*"; }

MIR=/root/loop_i2/mirror_r2.json
say "waiting for round 2's screen ($MIR)"
for _ in $(seq 1 480); do
  [ -s "$MIR" ] && break
  pgrep -f "[d]agger_loop_i2.sh" > /dev/null || { say "the loop exited before the screen finished"; exit 1; }
  sleep 60
done
[ -s "$MIR" ] || { say "STOP: the screen never completed"; exit 1; }
sleep 20
say "screen is up"

pkill -f "[d]agger_loop_i2.sh"; sleep 2
pkill -f "[m]irror_match.py"; pkill -f "[c]ollect_dagger.py"; sleep 8
say "loop paused"

cd /root/ptcg/repo
MODEL=/root/out/i2_r1 START_ROUND=2 DEADLINE_H=96 \
  setsid nohup bash tools/dagger_loop_i2b.sh > /dev/null 2>&1 &
sleep 10
say "tuned loop resumed at round 2 (bsz 32 x accum 1), pid $(pgrep -f '[d]agger_loop_i2b.sh' | head -1)"
