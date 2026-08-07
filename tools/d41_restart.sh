#!/usr/bin/env bash
# Restart the DeBERTa loop at round 4 so it picks up the fixed target ladder.
#
# WHY A RESTART IS REQUIRED. bash reads a script incrementally from an open fd, so a running loop
# keeps executing the bytes of the inode it started with. `mv` swaps the inode -- which is what
# makes the swap SAFE -- and is exactly why the running process does not see the change. Nothing
# short of restarting the loop applies it.
#
# WHAT THE FIX IS. The target ladder read `d[k]['verdict'] == 'WORSE'` out of the screen file. In
# mirror mode mirror_match runs the SPRT on discordant PAIRS (11-13 per deck against 40 games),
# and at that sample size the non-inferiority boundary is unreachable, so every deck comes back
# "undecided" and the WORSE tier is ALWAYS empty. This loop has screened in mirror mode since
# round 1, so it fell through to the wide below45 tier every round -- 20 decks at 72 games -- and
# never once used the one mechanism whose effect has actually been observed (gte went WORSE 4->0
# across r2->r3 with its mean flat at 40.2->40.1: the tail moved, the average did not).
# Recomputed from the raw w/l, round 3's screen has 7 WORSE decks.
#
# WAIT FOR THE MERGED SCREEN, THEN KILL. mirror_r4.json is written once, after every shard is
# read. Restarting at START_ROUND=4 re-uses it (screen_model returns early when the merged file
# exists), so the ~2 hours already spent screening d41_r3 are not thrown away, and PREV re-seeds
# from mirror_r3.json so the paired line survives the restart.
#
# WAIT=screen  wait for round $ROUND's merged screen, then restart at $ROUND re-using it. Use
#              when the change affects what happens AFTER the screen (targets, collect, train).
# WAIT=trained wait for round $((ROUND-1))'s checkpoint, then restart at $ROUND before its screen
#              starts. Use when the change affects the SCREEN itself -- otherwise the round-5
#              screen would run under the old sharding and the fix would slip another round.
#
#   WAIT=trained ROUND=5 nohup setsid bash tools/d41_restart.sh > /root/d41_restart5.log 2>&1 &
set -u
STATE=${STATE:-/root/loop_deberta41}
ROUND=${ROUND:-4}
MERGED=$STATE/mirror_r$ROUND.json
SCRIPT=${SCRIPT:-/root/d41_run.sh}
SRC=${SRC:-/root/ptcg/repo/tools/dagger_loop8.sh}
TIMEOUT_MIN=${TIMEOUT_MIN:-480}
WAIT=${WAIT:-screen}
OUTSTEM=${OUTSTEM:-/root/out/d41_r}
FROM=${FROM:-$OUTSTEM$((ROUND-1))}

say() { echo "[d41restart $(date -u +%m-%d_%H:%M:%S)] $*"; }

# Stage the new script under a temp name and mv it into place: scp/cp onto the SAME inode under a
# running bash resumes it at a byte offset that no longer means what it did.
cp "$SRC" "$SCRIPT.new" || { say "cannot stage $SRC"; exit 1; }
bash -n "$SCRIPT.new" || { say "staged script does not parse -- refusing"; exit 1; }
grep -q "from mirror_match import sprt" "$SCRIPT.new" \
  || { say "staged script does NOT contain the ladder fix -- refusing"; exit 1; }
if [ "$WAIT" = trained ]; then
  grep -q "LPT\|least work so far" "$SCRIPT.new" \
    || { say "staged script has no cost-balanced sharding -- refusing"; exit 1; }
fi
# The stage happens when the watcher ARMS, not when it swaps, so a fix written to $SRC after
# arming would be silently left behind. These greps are what catches that.
grep -q "base taper step" "$SCRIPT.new" \
  || { say "staged script has no base taper -- refusing"; exit 1; }
grep -q 'max-samples "\$RTOTAL"' "$SCRIPT.new" \
  || { say "staged script still runs to the deadline, so the taper would only add epochs"; exit 1; }
grep -q -- '--accum "\$ACCUM"' "$SCRIPT.new" \
  || { say "staged script lacks the ACCUM param (update starvation stays) -- refusing"; exit 1; }
grep -q -- '--pilot-decks "\$PILOT_DECKS"' "$SCRIPT.new" \
  || { say "staged script lacks the pilot-11 filter -- refusing"; exit 1; }
say "staged the fixed loop at $SCRIPT.new"

# NOT a test on $FROM/model.safetensors: the loop copies the previous checkpoint into $OUT at the
# START of the round, so that file exists from minute one and the wait would fire instantly. The
# log line is written only after training returns AND the RESUME check has passed.
ready() {
  if [ "$WAIT" = trained ]; then
    grep -aq "round $((ROUND - 1)) done -> $FROM" "$STATE/loop.log"
  else
    [ -s "$MERGED" ]
  fi
}
if [ "$WAIT" = trained ]; then
  say "waiting for round $((ROUND - 1)) to finish training into $FROM"
else
  say "waiting for $MERGED (round-$ROUND screen)"
fi
waited=0
while ! ready; do
  if ! pgrep -f "$(basename "$SCRIPT")" > /dev/null 2>&1; then
    say "the loop is already gone; applying the swap and starting round $ROUND"
    break
  fi
  if [ "$waited" -ge "$TIMEOUT_MIN" ]; then
    say "TIMEOUT after ${waited} min -- NOT killing a phase that is merely slow. See $STATE/loop.log"
    exit 1
  fi
  sleep 60
  waited=$((waited + 1))
done
say "ready after ${waited} min"

pkill -f "$(basename "$SCRIPT")" 2>/dev/null || true
sleep 3
# The collect step is a separate process that survives its parent. Kill it by the tag only this
# loop uses -- never a bare python3, which would take the Q-label generator with it.
pkill -f "collect_dagger.py" 2>/dev/null || true
pkill -f "mirror_match.py .*mirror_r$ROUND" 2>/dev/null || true
sleep 3
say "remaining loop procs: $(pgrep -c -f "$(basename "$SCRIPT")" 2>/dev/null || echo 0)"

mv "$SCRIPT.new" "$SCRIPT" || { say "swap FAILED"; exit 1; }
say "swapped $SCRIPT to the fixed ladder"

cd /root/ptcg/repo || exit 1
# DEADLINE_H is measured from the loop's own start, so a restart would silently hand it a fresh
# 72 hours. The original run began 08-06 03:53 UTC with 72h; what is left is what it gets.
LEFT=$(python3 -c "
import datetime as dt
end = dt.datetime(2026, 8, 9, 3, 53, tzinfo=dt.timezone.utc)
print(max(1, int((end - dt.datetime.now(dt.timezone.utc)).total_seconds() // 3600)))")
say "relaunching at round $ROUND from $FROM with DEADLINE_H=$LEFT"
KIND=deberta41 OUTSTEM=/root/out/d41_r START_ROUND=$ROUND MODEL=$FROM \
  DEADLINE_H=$LEFT VALUED=/root/ptcg/repo/data/rerank/v41_attach.jsonl.gz VALUED_FRAC=0.05 \
  setsid nohup bash "$SCRIPT" >> /root/d41_start.log 2>&1 < /dev/null &
sleep 20
say "loop procs now: $(pgrep -c -f "$(basename "$SCRIPT")" 2>/dev/null || echo 0)"
grep -aE "TIER=|targets:" "$STATE/loop.log" | tail -2
say "done"
