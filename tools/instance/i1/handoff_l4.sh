#!/bin/bash
# Hand instance1 from dagger_loop3 to dagger_loop4 at the round-3 boundary.
#
# loop3's round 2 trained on a single deck (ns_zoroark) and a 67,220-row mix -- half of round 1's
# and carrying only 3,361 of the 21,600 valued attach records -- because loop3 ties the round size
# to the DAgger count and stops the target ladder at the first non-empty tier. loop4 fixes both.
#
# The handoff waits for loop3 to publish round 3's SCREEN (that is the readout on round 2, and it
# costs 30 minutes to produce), then stops loop3 before it can spend 5 hours training another
# under-powered round, and restarts loop4 at round 3 with that screen pre-seeded so nothing is
# recomputed.
set -u
LOG=/root/handoff_l4.log
exec >> "$LOG" 2>&1
say() { echo "[handoff $(date -u +%m-%d_%H:%M:%S)] $*"; }

MIR3=/root/loop_rerank3/mirror_r3.json
say "waiting for $MIR3"
for _ in $(seq 1 720); do            # 12 hours of 60s polls
  [ -s "$MIR3" ] && break
  pgrep -f "[d]agger_loop3.sh" > /dev/null || { say "loop3 exited before round 3 -- stopping"; exit 1; }
  sleep 60
done
[ -s "$MIR3" ] || { say "round 3 screen never appeared -- stopping"; exit 1; }
sleep 20
say "round 3 screen is up"

pkill -f "[d]agger_loop3.sh"
sleep 2
pkill -f "[c]ollect_dagger.py"
pkill -f "[m]irror_match.py"
sleep 5
say "loop3 stopped (remaining: $(pgrep -fc '[d]agger_loop3.sh|[c]ollect_dagger.py' || echo 0))"

mkdir -p /root/loop_rerank4
cp "$MIR3" /root/loop_rerank4/mirror_r3.json
cp /root/loop_rerank3/history.tsv /root/loop_rerank4/history_l3.tsv 2>/dev/null || true

cd /root/ptcg/repo
MODEL=/root/out/l3_r2 START_ROUND=3 DEADLINE_H=48 \
  nohup bash tools/dagger_loop4.sh > /dev/null 2>&1 &
sleep 10
say "loop4 started from /root/out/l3_r2 at round 3 (pid $(pgrep -f '[d]agger_loop4.sh' | head -1))"
