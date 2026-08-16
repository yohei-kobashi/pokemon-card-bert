#!/bin/bash
# Hand loop5 -> loop6 at the round-1/round-2 boundary, so the targeting change takes effect from
# the next round rather than mid-round.
#
# The running script is NOT edited in place: bash reads a script incrementally by byte offset, so
# rewriting a file a loop is executing can drop it into the middle of a different statement.
#
# The trigger is the loop's own "round 1 done" line, not the presence of the checkpoint: loop5
# COPIES the previous checkpoint into $OUT before training starts, so model.safetensors exists
# from the first minute and would fire this immediately.
set -u
LOG=/root/handoff_l6.log
exec >> "$LOG" 2>&1
say() { echo "[h6 $(date -u +%m-%d_%H:%M:%S)] $*"; }

say "waiting for loop5 to finish round 1"
for _ in $(seq 1 900); do
  grep -q "round 1 done -> /root/out/l5_r1" /root/loop_rerank5/loop.log 2>/dev/null && break
  pgrep -f "[d]agger_loop5.sh" > /dev/null || { say "loop5 exited before round 1 finished"; exit 1; }
  sleep 60
done
grep -q "round 1 done -> /root/out/l5_r1" /root/loop_rerank5/loop.log 2>/dev/null \
  || { say "STOP: round 1 never completed"; exit 1; }
sleep 20
say "round 1 done"

pkill -f "[d]agger_loop5.sh"; sleep 2
pkill -f "[m]irror_match.py"; pkill -f "[c]ollect_dagger.py"; sleep 5
say "loop5 stopped"

mkdir -p /root/loop_rerank6
cp /root/loop_rerank5/history.tsv /root/loop_rerank6/history_l5.tsv 2>/dev/null || true

cd /root/ptcg/repo
KIND=rerank6 MODEL=/root/out/l5_r1 START_ROUND=2 TOTAL=1200000 LR=1e-5 DEADLINE_H=96 \
  setsid nohup bash tools/dagger_loop6.sh > /dev/null 2>&1 &
sleep 10
say "loop6 started from /root/out/l5_r1 at round 2 (pid $(pgrep -f '[d]agger_loop6.sh' | head -1))"
