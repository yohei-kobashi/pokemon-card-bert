#!/bin/bash
# Swap dagger_loop6.sh for dagger_loop7.sh (seeded mirror screen + anchored collection) at the
# round-4 boundary.
#
# Not an in-place edit: bash reads a running script by BYTE OFFSET, so editing loop6 while it
# executes would resume mid-token in a file that no longer matches. Wait for its own "round N
# done" line -- that line specifically, not the checkpoint directory appearing, because the loop
# creates $OUT before it has finished writing to it.
set -u
STATE=/root/loop_rerank6
LOG=$STATE/loop.log
HLOG=/root/handoff_l7.log
LOOP_PID=${LOOP_PID:?set LOOP_PID to the running loop6 pid}
DONE_ROUND=${DONE_ROUND:-4}
NEXT_ROUND=$((DONE_ROUND + 1))
NEXT_MODEL=/root/out/l6_r$DONE_ROUND
MAX_WAIT_H=${MAX_WAIT_H:-12}

exec >> "$HLOG" 2>&1
say() { echo "[handoff $(date -u +%m-%d_%H:%M:%S)] $*"; }

say "waiting for 'round $DONE_ROUND done' in $LOG (loop6 pid $LOOP_PID, giving it ${MAX_WAIT_H}h)"
T0=$(date +%s)
while :; do
  if grep -aq "round $DONE_ROUND done -> $NEXT_MODEL" "$LOG"; then say "round $DONE_ROUND finished"; break; fi
  if ! kill -0 "$LOOP_PID" 2>/dev/null; then
    say "ABORT: loop6 (pid $LOOP_PID) exited without finishing round $DONE_ROUND -- last lines:"
    tr "\r" "\n" < "$LOG" | grep -av "^$" | tail -6
    exit 1
  fi
  if [ $(( ($(date +%s) - T0) / 3600 )) -ge "$MAX_WAIT_H" ]; then
    say "ABORT: ${MAX_WAIT_H}h passed and round $DONE_ROUND has not finished"; exit 1
  fi
  sleep 30
done

kill "$LOOP_PID" 2>/dev/null
sleep 3
pkill -f "tools/mirror_match.py" 2>/dev/null
pkill -f "tools/collect_dagger.py" 2>/dev/null
sleep 5
if pgrep -f "dagger_loop6.sh" > /dev/null; then
  say "loop6 did not stop on SIGTERM -- sending SIGKILL"; pkill -9 -f "dagger_loop6.sh" 2>/dev/null; sleep 3
fi
say "stopped loop6; remaining loop processes:"
ps -eo pid,etime,cmd | grep -aE "dagger_loop|mirror_match|collect_dagger" | grep -av grep

[ -d "$NEXT_MODEL" ] || { say "ABORT: $NEXT_MODEL missing"; exit 1; }

rm -f "$STATE/mirror_r$NEXT_ROUND".*.json "$STATE/mirror_r$NEXT_ROUND.json" "$STATE/screen_r$NEXT_ROUND".*.log
say "cleared any partial round-$NEXT_ROUND screen artifacts"

cd /root/ptcg/repo
MODEL=$NEXT_MODEL START_ROUND=$NEXT_ROUND DEADLINE_H=${DEADLINE_H:-96} \
  setsid nohup bash tools/dagger_loop7.sh < /dev/null > /root/l7_boot.log 2>&1 &
sleep 5
say "started loop7 from $NEXT_MODEL at round $NEXT_ROUND:"
ps -eo pid,etime,cmd | grep -a "[d]agger_loop7.sh"
tail -4 "$LOG"
