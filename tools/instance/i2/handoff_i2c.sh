#!/bin/bash
# Swap dagger_loop_i2b.sh for dagger_loop_i2c.sh at the round-2 boundary.
#
# Not an in-place edit: bash reads a running script by BYTE OFFSET, so editing i2b while it
# executes would make it resume mid-token in a file that no longer matches. Wait for its own
# "round N done" line instead -- and specifically that line, not the checkpoint directory
# appearing, because the loop creates $OUT before it has finished writing to it.
set -u
STATE=/root/loop_i2
LOG=$STATE/loop.log
HLOG=/root/handoff_i2c.log
LOOP_PID=${LOOP_PID:?set LOOP_PID to the running i2b pid}
DONE_ROUND=${DONE_ROUND:-2}
NEXT_ROUND=$((DONE_ROUND + 1))
NEXT_MODEL=/root/out/i2_r$DONE_ROUND
MAX_WAIT_H=${MAX_WAIT_H:-12}

exec >> "$HLOG" 2>&1
say() { echo "[handoff $(date -u +%m-%d_%H:%M:%S)] $*"; }

say "waiting for 'round $DONE_ROUND done' in $LOG (i2b pid $LOOP_PID, giving it ${MAX_WAIT_H}h)"
T0=$(date +%s)
while :; do
  if grep -aq "round $DONE_ROUND done -> $NEXT_MODEL" "$LOG"; then
    say "round $DONE_ROUND finished"
    break
  fi
  if ! kill -0 "$LOOP_PID" 2>/dev/null; then
    say "ABORT: i2b (pid $LOOP_PID) exited without finishing round $DONE_ROUND -- last lines:"
    tr '\r' '\n' < "$LOG" | grep -av "^$" | tail -6
    exit 1
  fi
  if [ $(( ($(date +%s) - T0) / 3600 )) -ge "$MAX_WAIT_H" ]; then
    say "ABORT: ${MAX_WAIT_H}h passed and round $DONE_ROUND has not finished"
    exit 1
  fi
  sleep 30
done

# Stop the parent FIRST so it cannot launch anything else, then its screen children. The loop is
# in `wait` at this point in the worst case, and killing the parent leaves those orphaned.
kill "$LOOP_PID" 2>/dev/null
sleep 3
pkill -f "tools/mirror_match.py" 2>/dev/null
pkill -f "tools/collect_dagger.py" 2>/dev/null
sleep 5
if pgrep -f "dagger_loop_i2b.sh" > /dev/null; then
  say "i2b did not stop on SIGTERM -- sending SIGKILL"
  pkill -9 -f "dagger_loop_i2b.sh" 2>/dev/null
  sleep 3
fi
say "stopped i2b; remaining loop/scorer processes:"
ps -eo pid,etime,cmd | grep -aE "dagger_loop|mirror_match|collect_dagger|sft_teacher" | grep -av grep

[ -f "$NEXT_MODEL/domain_embeddings.pt" ] \
  || { say "ABORT: $NEXT_MODEL has no domain_embeddings.pt -- the round did not save cleanly"; exit 1; }

# Round $NEXT_ROUND may have got as far as launching screen shards in the seconds before the kill.
# A half-written shard file would be read as a finished one, so clear them; the merged
# mirror_r$NEXT_ROUND.json is what i2c checks for and must not exist either.
rm -f "$STATE/mirror_r$NEXT_ROUND".*.json "$STATE/mirror_r$NEXT_ROUND.json" \
      "$STATE/screen_r$NEXT_ROUND".*.log
say "cleared any partial round-$NEXT_ROUND screen artifacts"

cd /root/ptcg/repo
MODEL=$NEXT_MODEL START_ROUND=$NEXT_ROUND DEADLINE_H=${DEADLINE_H:-96} \
  setsid nohup bash tools/dagger_loop_i2c.sh < /dev/null > /root/i2c_boot.log 2>&1 &
sleep 5
say "started i2c from $NEXT_MODEL at round $NEXT_ROUND:"
ps -eo pid,etime,cmd | grep -a "[d]agger_loop_i2c.sh"
tail -4 "$LOG"
