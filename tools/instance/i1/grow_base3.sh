#!/bin/bash
# Grow the base pool 2.5x, because warm-start changed what the pool is FOR.
#
# Under from-scratch training the pool only had to be representative: each round drew a sample,
# trained once, and threw the weights away, so re-drawing the same rows next round cost nothing.
# dagger_loop5 continues the previous checkpoint, so exposure now ACCUMULATES in the weights and
# a repeated row is an extra epoch on data the model has already fitted.
#
# At 1,080,000 base rows per round against 5,733,620, a round consumes 18.8% of the pool:
#
#     round      fresh this round     pool seen so far
#       1              100.0%               18.8%
#       3               65.9%               46.5%
#       5               43.4%               64.8%
#      10               15.3%               87.6%
#
# By round 5 more than half of what the model sees is data it has already trained on. At 14.3M
# the same rounds are 73.1% and 49.4% fresh.
#
# THREE BATCHES, each its own tag. The cg RNG is not seedable (gen_selfplay's own docstring says
# so), which is exactly why re-running it produces genuinely different games rather than a copy.
#
# 28 workers, not 61: the box's quota is 61.4 cores and dagger_loop5 needs CPU for screening and
# collection. This waits for the training step before starting, so the DAgger collection -- which
# is latency-critical for the round -- runs uncontended.
set -u
REPO=/root/ptcg/repo
BASE=$REPO/data/rerank/v40_base.jsonl.gz
LOG=/root/grow_base3.log
cd "$REPO"
exec >> "$LOG" 2>&1
say() { echo "[grow3 $(date -u +%m-%d_%H:%M:%S)] $*"; }

say "waiting for the round's training to start (so collection runs uncontended)"
for _ in $(seq 1 120); do
  pgrep -f "[t]rain_rerank.py" > /dev/null && break
  sleep 30
done
say "training is up; starting generation"

for TAG in grow_a grow_b grow_c; do
  FREE=$(df -Pk /root | awk 'NR==2 {print int($4/1048576)}')
  say "=== $TAG | $FREE GiB free ==="
  [ "$FREE" -lt 12 ] && { say "STOP: under 12 GiB free, refusing to generate"; exit 1; }

  CUDA_VISIBLE_DEVICES= nice -n 15 python3 tools/gen_selfplay.py --games 12 --workers 28 \
      --tag "$TAG" 2>&1 | tail -3 || { say "STOP: generation failed for $TAG"; exit 1; }

  CUDA_VISIBLE_DEVICES= nice -n 15 python3 tools/build_rerank.py --tag "$TAG" --pfmt current \
      --label heuristic --sides both --workers 28 2>&1 | tail -3 \
      || { say "STOP: build failed for $TAG"; exit 1; }

  NEW=$REPO/data/rerank/$TAG.rerank.jsonl.gz
  [ -s "$NEW" ] || { say "STOP: $NEW is missing"; exit 1; }

  # The format check is the point of failure that would be invisible otherwise: rows rendered in
  # a different prompt format read fine, train fine, and only show up as a worse win rate.
  python3 - "$NEW" "$BASE" <<'PY' || { say "STOP: $TAG is not in the pool's prompt format"; exit 1; }
import gzip, json, sys
def profile(path, cap=3000):
    n = roles = facts = 0
    for line in gzip.open(path, "rt"):
        s = json.loads(line).get("state") or ""
        if " :: " not in s:
            continue
        n += 1
        roles += ("DECK win[" in s or "DECK eng[" in s or "DECK line[" in s)
        facts += ("need:" in s)
        if n >= cap:
            break
    return n, roles / n, facts / n
a = profile(sys.argv[1]); b = profile(sys.argv[2])
print("[fmt] new  n=%d roles=%.2f need=%.2f" % a)
print("[fmt] pool n=%d roles=%.2f need=%.2f" % b)
raise SystemExit(0 if all(abs(x - y) < 0.05 for x, y in zip(a[1:], b[1:])) else 1)
PY

  # An mv is atomic within a filesystem: a mix already reading the old pool keeps its inode and
  # finishes on the old data; the next round opens the new one. No round sees a half-written pool.
  cat "$BASE" "$NEW" > "$REPO/data/rerank/v40_base_ext.jsonl.gz.part" \
      || { say "STOP: concatenation failed"; exit 1; }
  python3 - "$REPO/data/rerank/v40_base_ext.jsonl.gz.part" <<'PY' || { say "STOP: the extended pool is not readable end to end"; exit 1; }
import gzip, json, sys
n = 0
for line in gzip.open(sys.argv[1], "rt"):
    json.loads(line); n += 1
print("[check] extended pool reads clean: %d records" % n)
PY
  mv "$REPO/data/rerank/v40_base_ext.jsonl.gz.part" "$BASE"
  rm -f "$NEW"
  rm -rf "$REPO/data/selfplay/$TAG"
  say "SWAPPED after $TAG: pool is now $(zcat "$BASE" | wc -l) records"
done
say "GROW DONE"
