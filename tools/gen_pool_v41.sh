#!/usr/bin/env bash
# Build a PURE v41 base pool, continuously, with no training to wait for.
#
# pool_daemon.sh only generated while train_rerank.py held the GPU -- correct when the box was
# also training, wasteful now that instance1's reranker loop is stopped (rounds 4->8 measured
# -0.71pt +- 1.33 paired, i.e. flat). With the GPU idle the 61 effective cores do a 23k-game
# batch in ~5 minutes, so the pool can be rebuilt in an hour instead of accumulated over days.
#
# It writes a NEW FILE rather than appending to v40_base. That is the whole point: no mixed pool,
# so no pruning step and no round that trains on two prompt formats at once.
#
# SFT batches are dropped in $PENDING for tools/ship_pool_v41.sh to move to instance2
# separately. Shipping is bandwidth-bound (measured 46-240 KB/s, ~70 min per batch) and
# generation is not, so the two must not block each other.
#
#   TARGET_ROWS=10000000 nohup bash tools/gen_pool_v41.sh > /root/gen_v41.log 2>&1 &
set -u
REPO=${REPO:-/root/ptcg/repo}
BASE=${BASE:-$REPO/data/rerank/v41_base.jsonl.gz}
PENDING=${PENDING:-$REPO/data/sft/v41_pending}
CNT=${CNT:-/root/gen_v41.count}
TARGET_ROWS=${TARGET_ROWS:-10000000}
GAMES=${GAMES:-12}
WORKERS=${WORKERS:-28}
MIN_FREE_GIB=${MIN_FREE_GIB:-12}

cd "$REPO"
mkdir -p "$PENDING" "$(dirname "$BASE")"
[ -f "$CNT" ] || echo 0 > "$CNT"
say() { echo "[genv41 $(date -u +%m-%d_%H:%M:%S)] $*"; }

rows() { [ -s "$1" ] && zcat "$1" | wc -l || echo 0; }

say "target $TARGET_ROWS rows -> $BASE"
while true; do
  HAVE=$(rows "$BASE")
  if [ "$HAVE" -ge "$TARGET_ROWS" ]; then
    say "reached $HAVE rows -- done"; break
  fi
  FREE=$(df -Pk "$REPO" | awk 'NR==2 {print int($4/1048576)}')
  if [ "$FREE" -lt "$MIN_FREE_GIB" ]; then
    say "only $FREE GiB free -- waiting for the shipper to drain $PENDING"; sleep 300; continue
  fi

  N=$(( $(cat "$CNT") + 1 )); echo "$N" > "$CNT"
  TAG=v41_$N
  say "=== $TAG | have $HAVE rows | $FREE GiB free ==="

  # --keep-blob is REQUIRED: v41 decodes the engine's hidden state out of
  # obs["search_begin_input"], and without it every hidden fact renders as absent -- silently,
  # because they are all optional.
  CUDA_VISIBLE_DEVICES= nice -n 5 python3 tools/gen_selfplay.py --games "$GAMES" \
      --workers "$WORKERS" --keep-blob --tag "$TAG" 2>&1 | tail -2 \
      || { say "generation failed"; sleep 60; continue; }
  CUDA_VISIBLE_DEVICES= nice -n 5 python3 tools/build_rerank.py --tag "$TAG" --pfmt v41 \
      --label heuristic --sides both --workers "$WORKERS" 2>&1 | tail -2 \
      || { say "build failed"; rm -rf "$REPO/data/selfplay/$TAG"; sleep 60; continue; }

  NEW=$REPO/data/rerank/$TAG.rerank.jsonl.gz
  [ -s "$NEW" ] || { say "$NEW missing"; rm -rf "$REPO/data/selfplay/$TAG"; continue; }

  # Every row must carry pfmt=v41. A batch built from a tag generated without the blob still
  # stamps v41 but renders v40 content, and nothing downstream would notice.
  python3 - "$NEW" <<'PY' || { say "stamp check FAILED -- batch discarded"; rm -f "$NEW"; rm -rf "$REPO/data/selfplay/$TAG"; continue; }
import gzip, json, sys
n = bad = hits = 0
for line in gzip.open(sys.argv[1], "rt"):
    d = json.loads(line); n += 1
    if d.get("pfmt") != "v41": bad += 1
    s = d.get("state", "")
    if " d:" in s or " d~" in s or " n:" in s or " thr" in s or " tk:" in s: hits += 1
if bad or not n: raise SystemExit("pfmt stamp missing on %d/%d rows" % (bad, n))
if hits * 100 < n * 5:
    raise SystemExit("only %d/%d rows carry a v41 fact -- was the tag built with --keep-blob?"
                     % (hits, n))
print("[check] %d rows, all pfmt=v41, %d (%.1f%%) carry a v41 fact" % (n, hits, 100.0*hits/n))
PY

  if [ -s "$BASE" ]; then cat "$BASE" "$NEW" > "$BASE.part"; else cp "$NEW" "$BASE.part"; fi
  python3 - "$BASE.part" <<'PY' || { say "append failed -- pool untouched"; rm -f "$BASE.part"; continue; }
import gzip, json, sys
n = 0
for line in gzip.open(sys.argv[1], "rt"):
    json.loads(line); n += 1
print("[check] pool reads clean: %d records" % n)
PY
  mv "$BASE.part" "$BASE"

  # decoder schema for instance2; the shipper picks these up
  rm -f /tmp/${TAG}_s*.gz
  for i in 0 1 2 3 4 5 6 7; do
    nice -n 5 python3 tools/rerank_to_sft.py --inp "$NEW" --out /tmp/${TAG}_s$i.gz \
        --shard $i --nshards 8 > /dev/null 2>&1 &
  done
  wait
  cat /tmp/${TAG}_s[0-7].gz > "$PENDING/$TAG.sft.jsonl.gz.part" \
      && mv "$PENDING/$TAG.sft.jsonl.gz.part" "$PENDING/$TAG.sft.jsonl.gz"
  rm -f /tmp/${TAG}_s*.gz

  rm -f "$NEW"
  rm -rf "$REPO/data/selfplay/$TAG"
  say "pool now $(rows "$BASE") rows | pending sft batches: $(ls -1 "$PENDING" | wc -l)"
done
say "generation finished"
