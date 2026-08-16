#!/usr/bin/env bash
# Wait for the RL loop's r8 gate to release the GPU, then run rank_probe on the 8-playout
# branch data. The gate fans out to 21 processes and fills the card (24,080 of 24,564 MiB
# measured), so starting alongside it is a guaranteed OOM -- and an OOM would kill the GATE,
# not just this job. Hence: launch only when no eval process is alive AND the card is mostly
# free, and re-check immediately before exec.
#
# The r12 gate (~11:00 UTC) is the deadline. rank_probe at 2 epochs over ~45k training points
# is ~80 min, so it must start by ~09:30 to be clear of it; past that, do not start.
set -u
LOG=/root/auto_decision.log
say () { echo "[$(date -u '+%F %T')] $*" | tee -a "$LOG"; }

DATA=/root/out/branch8.jsonl.gz
CKPT=/root/out/rlDL/A_r6_policy      # the same checkpoint rlDL2 started RL from
OUT=/root/rank_probe.log
DEADLINE=$(date -u -d '09:30' +%s)   # today, UTC

say "=== auto_rank_probe armed (waiting for the r8 gate to finish) ==="

while :; do
  n=$(pgrep -cf eval_rerank || true)
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  now=$(date -u +%s)
  if [ "$now" -gt "$DEADLINE" ]; then
    say "DECISION: past 09:30 UTC and the GPU never freed -- NOT starting (the r12 gate needs the card)"
    exit 1
  fi
  if [ "${n:-1}" -eq 0 ] && [ "${free:-0}" -ge 12000 ]; then
    break
  fi
  sleep 60
done

sleep 30                              # let the gate's processes fully release
n=$(pgrep -cf eval_rerank || true)
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "${n:-1}" -ne 0 ] || [ "${free:-0}" -lt 12000 ]; then
  say "DECISION: a gate started during the settle window (procs=$n free=${free}MiB) -- NOT starting"
  exit 1
fi

say "DECISION: GPU free (${free} MiB). Running rank_probe."
say "  data  $DATA  (99,139 points, 59,974 usable after the degenerate filter)"
say "  from  $CKPT  (rlDL2's own RL start, so this is supervised-vs-RL from the SAME point)"
say "  the number to beat: the policy scores +0.0063 +/- 0.0058; the measured ceiling on"
say "  usable rows is +0.1638 +/- 0.0021 (+0.0991 if degenerate rows are counted as 0)"

cd /root/ptcg/repo || exit 1
python tools/rank_probe.py --rollouts "$DATA" --model "$CKPT" --epochs 2 \
    > "$OUT" 2>&1
rc=$?
say "rank_probe exited $rc; tail:"
grep -iE "metric|holdout|VERDICT|epoch|E\[|ceiling|chance" "$OUT" 2>/dev/null \
  | tail -20 | sed 's/^/    /' | tee -a "$LOG"
say "full output: $OUT"
say "NOTE: the --shuffle-control null run is NOT started here -- it is another ~80 min and"
say "would collide with the r12 gate. Run it after the RL loop finishes if the probe shows a gain."
say "=== auto_rank_probe done ==="
