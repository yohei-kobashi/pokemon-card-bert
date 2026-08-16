#!/usr/bin/env bash
# Let instance2's round-8 SCREEN finish, then stop the old loop before it trains.
#
# WHY. The screen currently running evaluates /root/out/i2_r7, which is the only way to learn
# whether round 7 moved anything -- worth the ~2 hours it costs. What follows it is ~9 more
# hours of collect + training on the A/B/C curriculum that docs/rl_stages_v2.md retires, and
# that is 10% of a 89-hour time box spent on a plan we have replaced.
#
# So: wait for the screen, take its result, stop.
#
# THE WAIT CONDITION IS THE MERGED FILE, NOT A LOG LINE. The loop prints "merged -> N decks"
# from a subshell and then immediately starts the target selection; keying on the text would
# race. mirror_r8.json is written once, atomically, after every shard is read.
#
#   nohup bash tools/i2_stop_after_screen.sh > /root/i2_stop.log 2>&1 &
set -u
STATE=${STATE:-/root/loop_i2_v41}
ROUND=${ROUND:-8}
MERGED=${MERGED:-$STATE/mirror_$ROUND.json}
MERGED=${MERGED:-$STATE/mirror_r$ROUND.json}
LOOP_PAT=${LOOP_PAT:-dagger_loop_i2}
TIMEOUT_MIN=${TIMEOUT_MIN:-240}

say() { echo "[stop $(date -u +%m-%d_%H:%M:%S)] $*"; }
MERGED=$STATE/mirror_r$ROUND.json
say "waiting for $MERGED (round-$ROUND screen), then stopping '$LOOP_PAT'"

waited=0
while [ ! -s "$MERGED" ]; do
  if ! pgrep -f "$LOOP_PAT" > /dev/null 2>&1; then
    say "the loop is already gone and $MERGED never appeared -- nothing to stop"
    exit 1
  fi
  if [ "$waited" -ge "$TIMEOUT_MIN" ]; then
    say "TIMEOUT after ${waited} min. NOT stopping the loop -- a screen that is merely slow"
    say "must not be killed by a watchdog. Inspect $STATE/loop.log."
    exit 1
  fi
  sleep 60
  waited=$((waited + 1))
done

say "screen merged after ${waited} min"
# Print the round's own verdict before killing anything, so the reason this round existed is
# in this log and not only in a file someone has to know to open.
python3 - "$MERGED" "$STATE/mirror_r$((ROUND - 1)).json" <<'PY' || true
import json, math, statistics as st, sys
d = json.load(open(sys.argv[1]))["decks"]
p = [v["p"] for v in d.values()]
print("[screen] round %s | decks %d | mean %.1f%% | median %.1f%% | below50 %d"
      % (sys.argv[1].split("_r")[-1][0], len(p), 100*st.mean(p), 100*st.median(p),
         sum(1 for x in p if x < .5)))
try:
    q = json.load(open(sys.argv[2]))["decks"]
except Exception:
    raise SystemExit(0)
both = sorted(set(d) & set(q))
if len(both) > 2:
    dd = [d[k]["p"] - q[k]["p"] for k in both]
    m = st.mean(dd); se = st.stdev(dd) / math.sqrt(len(dd))
    print("[paired vs previous round, %d decks] %+.4f +- %.4f  t %+.2f"
          % (len(both), m, se, m / se if se else 0.0))
PY

say "stopping the old curriculum loop"
pkill -f "$LOOP_PAT" || true
sleep 5
# The training step is a separate process and survives its parent; kill it by the tag only this
# loop uses, never a bare "python3", or the pricing job dies with it.
pkill -f "train_qwen.*i2_r" 2>/dev/null || true
pkill -f "mirror_match.py .*mirror_r$ROUND" 2>/dev/null || true
sleep 5
say "remaining loop procs: $(pgrep -c -f "$LOOP_PAT" 2>/dev/null || echo 0)"
say "GPU now: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader 2>/dev/null | head -1) free"
say "done -- tools/price_when_free.sh will pick the GPU up, then stage1_chain.sh follows"
