#!/bin/bash
# Keep both pools growing, one batch per training round, forever.
#
# WHY IT IS TIED TO THE TRAINING STEP. Generation wants 28 CPU workers; so does the DAgger
# collection, and the collection is on the round's critical path while the training is not (it
# holds the GPU and leaves the box's 61.4 cores idle). Generating during training is therefore
# free, and generating during collection is not.
#
# WHY ONE BATCH PER ROUND. instance1 draws 1,080,000 base rows per round. At 14.3M that is 7.5%
# of the pool, and one batch adds ~2.87M rows per ~6-hour round -- so the fraction the model has
# already seen stops climbing instead of reaching 85% by round 10. More often than that would
# outrun the link to instance2 (measured 46-240 KB/s; a batch is ~135 MB, so ~20-50 min) and
# would grow the pool faster than either box can consume it.
#
# THE TWO POOLS ARE DIFFERENT SCHEMAS. instance1 trains a cross-encoder on state/candidates/
# chosen; instance2 trains a decoder on prompt/target. rerank_to_sft.py converts, and the
# conversion is verified per record rather than assumed -- it was 5,733,620 direct, 0 fallbacks
# when the pools were first joined.
#
# Both appends are `mv`, which is atomic within a filesystem: a mix already reading the old pool
# keeps its inode and finishes on the old data; the next round opens the new one.
set -u
REPO=/root/ptcg/repo
BASE=$REPO/data/rerank/v40_base.jsonl.gz
I2BASE=/root/ptcg/repo/data/sft/v40_base_sft.jsonl.gz
CNT=/root/pool_daemon.count
LOG=/root/pool_daemon.log
cd "$REPO"
exec >> "$LOG" 2>&1
say() { echo "[pool $(date -u +%m-%d_%H:%M:%S)] $*"; }

[ -f "$CNT" ] || echo 0 > "$CNT"

say "waiting for grow_base3 to finish before taking over"
while pgrep -f "[g]row_base3.sh" > /dev/null; do sleep 60; done
say "daemon up"

while true; do
  # --- wait for a training step to be running -------------------------------------------
  while ! pgrep -f "[t]rain_rerank.py" > /dev/null; do sleep 60; done

  FREE=$(df -Pk /root | awk 'NR==2 {print int($4/1048576)}')
  if [ "$FREE" -lt 12 ]; then
    say "only $FREE GiB free -- skipping this round's batch"
    while pgrep -f "[t]rain_rerank.py" > /dev/null; do sleep 120; done
    continue
  fi

  N=$(( $(cat "$CNT") + 1 )); echo "$N" > "$CNT"
  TAG=pool_$N
  say "=== $TAG | $FREE GiB free ==="

  CUDA_VISIBLE_DEVICES= nice -n 15 python3 tools/gen_selfplay.py --games 12 --workers 28 \
      --tag "$TAG" --keep-blob 2>&1 | tail -2 || { say "generation failed; retrying next round"; continue; }
  CUDA_VISIBLE_DEVICES= nice -n 15 python3 tools/build_rerank.py --tag "$TAG" --pfmt v41 \
      --label heuristic --sides both --workers 28 2>&1 | tail -2 \
      || { say "build failed; retrying next round"; continue; }
  NEW=$REPO/data/rerank/$TAG.rerank.jsonl.gz
  [ -s "$NEW" ] || { say "$NEW missing; retrying next round"; continue; }

  # A batch rendered in a different prompt format reads fine, trains fine, and shows up only as
  # a worse win rate. Refuse it instead.
  python3 - "$NEW" "$BASE" <<'PY' || { say "$TAG is NOT in the pool's format -- discarded"; rm -f "$NEW"; continue; }
import gzip, json, sys
def profile(p, cap=3000):
    n = roles = facts = 0
    for line in gzip.open(p, "rt"):
        s = json.loads(line).get("state") or ""
        if " :: " not in s: continue
        n += 1
        roles += ("DECK win[" in s or "DECK eng[" in s or "DECK line[" in s)
        facts += ("need:" in s)
        if n >= cap: break
    return n, roles/n, facts/n
a, b = profile(sys.argv[1]), profile(sys.argv[2])
print("[fmt] new n=%d roles=%.2f need=%.2f | pool n=%d roles=%.2f need=%.2f" % (a+b))
raise SystemExit(0 if all(abs(x-y) < 0.05 for x, y in zip(a[1:], b[1:])) else 1)
PY

  # --- instance1 ------------------------------------------------------------------------
  cat "$BASE" "$NEW" > "$BASE.part" && python3 - "$BASE.part" <<'PY' && mv "$BASE.part" "$BASE"
import gzip, json, sys
n = 0
for line in gzip.open(sys.argv[1], "rt"):
    json.loads(line); n += 1
print("[check] i1 pool reads clean: %d records" % n)
PY
  [ -f "$BASE.part" ] && { say "i1 append failed -- pool untouched"; rm -f "$BASE.part"; }
  say "i1 pool now $(zcat "$BASE" | wc -l) records"

  # --- instance2: convert, ship, append -------------------------------------------------
  SFT=$REPO/data/sft/$TAG.sft.jsonl.gz
  rm -f /tmp/${TAG}_s*.gz
  for i in 0 1 2 3 4 5 6 7; do
    nice -n 15 python3 tools/rerank_to_sft.py --inp "$NEW" --out /tmp/${TAG}_s$i.gz \
        --shard $i --nshards 8 > /dev/null 2>&1 &
  done
  wait
  cat /tmp/${TAG}_s0.gz /tmp/${TAG}_s1.gz /tmp/${TAG}_s2.gz /tmp/${TAG}_s3.gz \
      /tmp/${TAG}_s4.gz /tmp/${TAG}_s5.gz /tmp/${TAG}_s6.gz /tmp/${TAG}_s7.gz > "$SFT"
  rm -f /tmp/${TAG}_s*.gz
  WANT=$(zcat "$SFT" | wc -l)
  say "converted to the decoder schema: $WANT rows -> shipping"

  # scp wants -P for the port, ssh wants -p; they are spelled out rather than shared through a
  # variable because getting that wrong fails silently into a retry loop.
  if scp -q -i /root/.ssh/id_i2 -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -P 19839 \
          "$SFT" root@175.155.64.145:/root/incoming_$TAG.gz; then
    # The row count is checked ON instance2. A truncated .gz decompresses fine up to the cut,
    # so a partial transfer would append a partial batch and nothing would error.
    ssh -i /root/.ssh/id_i2 -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -p 19839 \
        root@175.155.64.145 "bash -s" <<REMOTE
set -u
B=$I2BASE; INC=/root/incoming_$TAG.gz
FREE=\$(df -Pk /root | awk 'NR==2 {print int(\$4/1048576)}')
[ "\$FREE" -lt 20 ] && { echo "[i2] only \$FREE GiB free -- refusing"; exit 1; }
GOT=\$(zcat "\$INC" | wc -l) || { echo "[i2] the batch does not decompress"; exit 1; }
[ "\$GOT" = "$WANT" ] || { echo "[i2] got \$GOT rows, expected $WANT -- transfer incomplete"; exit 1; }
cat "\$B" "\$INC" > "\$B.part" || exit 1
python3 -c "
import gzip, json, sys
n=0
for line in gzip.open('\$B.part','rt'):
    json.loads(line); n+=1
print('[i2] pool reads clean: %d records' % n)
" || { rm -f "\$B.part"; exit 1; }
mv "\$B.part" "\$B"
rm -f "\$INC"
REMOTE
    if [ $? -eq 0 ]; then say "i2 pool extended with $TAG"; else say "i2 append FAILED for $TAG"; fi
  else
    say "transfer of $TAG FAILED -- instance1 kept it, instance2 skipped this batch"
  fi

  rm -f "$NEW" "$SFT"
  rm -rf "$REPO/data/selfplay/$TAG"

  # at most one batch per round
  while pgrep -f "[t]rain_rerank.py" > /dev/null; do sleep 120; done
  say "round's training ended; waiting for the next one"
done
