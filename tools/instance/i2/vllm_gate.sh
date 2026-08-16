#!/bin/bash
# When round 1's training finishes, take the card for the optimisation benchmark BEFORE the loop
# spends 3.8 hours on round 2's screen, then hand it back.
#
# The order matters: the loop's next step is the screen, which is the thing the benchmark might
# change how we run. Measuring after committing to it would be measuring too late. The benchmark
# is ~30 minutes against a 3.8-hour screen.
#
# The loop is restarted with the CURRENT scorer regardless of the result. Adopting anything here
# is a change to the harness that produces every number we trust, and is not made unattended on
# the strength of a number nobody has read yet.
set -u
LOG=/root/vllm_gate.log
exec >> "$LOG" 2>&1
say() { echo "[gate $(date -u +%m-%d_%H:%M:%S)] $*"; }

CKPT=/root/out/i2_r1
say "waiting for $CKPT/domain_embeddings.pt"
for _ in $(seq 1 900); do
  [ -f "$CKPT/domain_embeddings.pt" ] && break
  pgrep -f "[d]agger_loop_i2.sh" > /dev/null || { say "the loop exited before round 1 finished"; exit 1; }
  sleep 60
done
[ -f "$CKPT/domain_embeddings.pt" ] || { say "STOP: training never produced a checkpoint"; exit 1; }
sleep 30
say "round 1 checkpoint is up"

pkill -f "[d]agger_loop_i2.sh"; sleep 2
pkill -f "[m]irror_match.py"; pkill -f "[s]ft_teacher.py"; sleep 8
say "loop paused; GPU free"

bash /root/opt_bench.sh || say "WARNING: the optimisation benchmark exited non-zero"
say "benchmark done"

sleep 5
cd /root/ptcg/repo
MODEL=$CKPT START_ROUND=2 DEADLINE_H=96 setsid nohup bash tools/dagger_loop_i2.sh > /dev/null 2>&1 &
sleep 10
say "loop resumed at round 2 from $CKPT (pid $(pgrep -f '[d]agger_loop_i2.sh' | head -1))"
