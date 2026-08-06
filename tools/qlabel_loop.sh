#!/usr/bin/env bash
# instance1: keep making Q labels for the flagged (deck, kind) cells and push them to instance2.
#
# The two machines split by what they are good at. Branch-and-playout is pure CPU that
# parallelises perfectly, and instance1 has 61.4 effective cores against instance2's 13.44
# ([[vast-cpu-quotas]]); training is GPU and lives on instance2. Coupling them through a file
# drop rather than one orchestrator means a shutdown, a crash or a slow round on either side
# does not stall the other -- the same split that has been running gen/ship/train all week.
#
# Batches are ~2 MB (attach's 16,830 records compressed to 1.8 MB), so unlike the multi-GB base
# pool there is no reason to separate generation from shipping. It is one loop.
#
#   nohup bash tools/qlabel_loop.sh > /root/qlabel_loop.log 2>&1 &
set -u
REPO=${REPO:-/root/ptcg/repo}
TARGETS=${TARGETS:-/root/lm_targets_i2r6.json}
PER_DECK=${PER_DECK:-350}
PLAYOUTS=${PLAYOUTS:-16}
WORKERS=${WORKERS:-36}
DECK_SECONDS=${DECK_SECONDS:-2400}
GAMES=${GAMES:-600}
LOAD_CEIL=${LOAD_CEIL:-46}
CNT=${CNT:-/root/qlabel.count}
I2HOST=${I2HOST:-root@175.155.64.145}
I2PORT=${I2PORT:-19839}
I2KEY=${I2KEY:-/root/.ssh/id_i2}
I2DIR=${I2DIR:-/root/qlabel_in}
MAX_BATCHES=${MAX_BATCHES:-40}

say() { echo "[qlab $(date -u +%m-%d_%H:%M:%S)] $*"; }
cd "$REPO" || exit 1
[ -f "$CNT" ] || echo 0 > "$CNT"
[ -s "$TARGETS" ] || { say "no targets at $TARGETS"; exit 1; }

# Reachability is a WAIT, not a precondition. vast.ai rewrote instance2's authorized_keys under
# us on 08-06 and dropped instance1's key; the loop exited on the spot and nothing restarted it,
# so five hours of CPU produced batches that had nowhere to go. Generation is still useful while
# the link is down -- batches are kept locally and shipped when it comes back.
i2_up() { ssh -i "$I2KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no \
              -o ConnectTimeout=20 -p "$I2PORT" "$I2HOST" "mkdir -p $I2DIR" 2>/dev/null; }
n=0
while ! i2_up; do
  n=$((n + 1))
  [ $((n % 10)) -eq 1 ] && say "instance2 unreachable (${n} tries) -- check its authorized_keys"
  sleep 60
done

while :; do
  N=$(( $(cat "$CNT") + 1 ))
  [ "$N" -gt "$MAX_BATCHES" ] && { say "reached MAX_BATCHES=$MAX_BATCHES -- stopping"; break; }
  echo "$N" > "$CNT"

  # Yield to the DeBERTa loop. Its collect and screen phases are CPU-bound too, and a Q-label
  # batch that finishes an hour later costs nothing while a delayed training round costs a
  # round of the time box.
  while [ "$(awk '{print int($1)}' /proc/loadavg)" -gt "$LOAD_CEIL" ]; do
    say "load over $LOAD_CEIL -- holding"; sleep 300
  done

  TAG=qlab_$N
  OUT=$REPO/data/rerank/$TAG.jsonl.gz
  say "=== $TAG | per-deck $PER_DECK | $WORKERS workers ==="
  # Both env assignments must precede `nice`. `nice -n 5 PYTHONPATH=... python3` makes
  # nice try to EXECUTE "PYTHONPATH=cg-lib" as the program -- it fails instantly with
  # "No such file or directory" and the loop reads it as an empty batch.
  CUDA_VISIBLE_DEVICES= PYTHONPATH=cg-lib nice -n 5 python3 tools/qlabel_gen.py \
      --targets "$TARGETS" --per-deck "$PER_DECK" --playouts "$PLAYOUTS" \
      --workers "$WORKERS" --games "$GAMES" --deck-seconds "$DECK_SECONDS" \
      --seed $((9000 + N * 977)) --out "$OUT" 2>&1 | tail -14 \
      || { say "generation failed -- retrying in 10 min"; sleep 600; continue; }
  [ -s "$OUT" ] || { say "$OUT empty -- retrying in 10 min"; sleep 600; continue; }

  WANT=$(zcat "$OUT" | wc -l)
  say "shipping $TAG ($WANT records, $(du -h "$OUT" | cut -f1))"
  n=0
  while ! i2_up; do
    n=$((n + 1))
    [ $((n % 10)) -eq 1 ] && say "instance2 unreachable -- $TAG is held locally (${n} tries)"
    sleep 60
  done
  if scp -q -i "$I2KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -P "$I2PORT" \
        "$OUT" "$I2HOST:$I2DIR/.$TAG.part"; then
    # Rename REMOTELY after the copy so instance2's loop never sees a half-written file. A
    # consumer globbing the directory would otherwise pick up a truncated batch, which
    # decompresses cleanly up to the cut and looks like a small but valid one.
    ssh -i "$I2KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -p "$I2PORT" "$I2HOST" \
        "GOT=\$(zcat $I2DIR/.$TAG.part | wc -l); [ \"\$GOT\" = \"$WANT\" ] \
         && mv $I2DIR/.$TAG.part $I2DIR/$TAG.jsonl.gz \
         || { echo '[i2] got '\$GOT' of $WANT -- discarding'; rm -f $I2DIR/.$TAG.part; exit 1; }" \
      && { say "instance2 has $TAG"; rm -f "$OUT"; } || say "verify FAILED for $TAG -- kept locally"
  else
    say "transfer FAILED for $TAG -- keeping the local copy"
  fi
  # `rm -f "$OUT"` used to run unconditionally, one line after the branch that says it is
  # keeping the local copy. The two failure paths deleted exactly what they claimed to save.
done
say "qlabel loop ended"
