#!/usr/bin/env bash
# Keep the v41 base pool CURRENT, without starving the loop that consumes it.
#
# gen_pool_v41.sh is a one-shot: it breaks out of its loop the moment the pool reaches
# TARGET_ROWS and the process exits. That is correct for a rebuild and wrong for a standing
# task -- the pool stopped growing at 09:01 UTC on 2026-08-05 and nothing said so.
#
# WHY A LOW RATE, NOT A REBUILD. The reranker has consumed 320,399 of the pool's 12,155,331
# rows -- 2.6%. Ten more loop rounds at --total 300000 draw about 3M. More rows from the SAME
# engine_v2 are not the bottleneck and never will be at this ratio. What DOES decay is
# freshness: every per-deck rule that lands changes the moves being imitated, and every deck
# added (ogerpon_mono and dudunsparce_box most recently) is absent until it is generated. So
# this trickles rather than floods.
#
# WHY IT WAITS. instance1 has 61.4 effective cores, not the 256 nproc claims
# ([[vast-cpu-quotas]]). The DAgger loop's collect and screen phases and attach_label are both
# CPU-bound; starting a 28-worker generator on top of them delays the collect, which delays the
# training, which is the thing the pool exists to feed. The generator therefore
#
#   1. waits for any named job to clear, then
#   2. runs at WORKERS=16, and
#   3. pauses whenever the 1-minute load average exceeds LOAD_CEIL.
#
# The batches it produces are picked up by the already-running tools/ship_pool_v41.sh and
# appended to instance2's decoder pool; nothing extra needs to be started for that.
#
#   nohup bash tools/gen_pool_v41_resume.sh > /root/gen_v41_resume.log 2>&1 &
set -u
REPO=${REPO:-/root/ptcg/repo}
WAIT_FOR=${WAIT_FOR:-attach_label}          # regex of processes to let finish first ("" = none)
WORKERS=${WORKERS:-16}
LOAD_CEIL=${LOAD_CEIL:-40}
TARGET_ROWS=${TARGET_ROWS:-40000000}        # far above anything a round will draw; the real
                                            # stop conditions are disk and being killed
MIN_FREE_GIB=${MIN_FREE_GIB:-14}
# Default the pairing to the 11 submission candidates (user 2026-08-08: keep growing the
# 11-deck SFT data). Explicit PAIR_WITH="" reverts to all-pairs generation.
PAIR_WITH=${PAIR_WITH-$(PYTHONPATH=$REPO/tools python3 -c "import rl_config; print(','.join(rl_config.STAGE_C_TARGETS))" 2>/dev/null || true)}
say() { echo "[resume $(date -u +%m-%d_%H:%M:%S)] $*"; }

if [ -n "$WAIT_FOR" ]; then
  while pgrep -f "$WAIT_FOR" > /dev/null 2>&1; do
    say "waiting for '$WAIT_FOR' to finish ($(pgrep -c -f "$WAIT_FOR") procs)"
    sleep 120
  done
  say "'$WAIT_FOR' is done"
fi

# Load-gated relaunch. gen_pool_v41.sh exits on TARGET_ROWS or on a disk shortage; either way
# this re-enters it, so a pause is a pause and not a stop.
while true; do
  L=$(awk '{print int($1)}' /proc/loadavg)
  if [ "$L" -gt "$LOAD_CEIL" ]; then
    say "load $L over ceiling $LOAD_CEIL -- holding"
    sleep 300; continue
  fi
  FREE=$(df -Pk "$REPO" | awk 'NR==2 {print int($4/1048576)}')
  if [ "$FREE" -lt "$MIN_FREE_GIB" ]; then
    say "only $FREE GiB free -- holding for the shipper to drain"
    sleep 600; continue
  fi
  say "generating (load $L, ${FREE} GiB free, $WORKERS workers)"
  TARGET_ROWS=$TARGET_ROWS WORKERS=$WORKERS MIN_FREE_GIB=$MIN_FREE_GIB PAIR_WITH=$PAIR_WITH \
      bash "$REPO/tools/gen_pool_v41.sh" >> /root/gen_v41.log 2>&1
  say "gen_pool_v41.sh returned -- re-checking in 5 min"
  sleep 300
done
