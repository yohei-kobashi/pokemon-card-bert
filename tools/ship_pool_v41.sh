#!/usr/bin/env bash
# Move finished v41 SFT batches from instance1 to instance2's pool, one at a time.
#
# Separate from tools/gen_pool_v41.sh on purpose. Generation is CPU-bound and takes ~5 min a
# batch; shipping is bandwidth-bound at a measured 46-240 KB/s, i.e. ~70 min for the same batch.
# Running them in one loop would throttle generation to the link's speed.
#
# The row count is verified ON INSTANCE2 before the append: a truncated .gz decompresses fine up
# to the cut, so a partial transfer would silently append a partial batch.
#
#   nohup bash tools/ship_pool_v41.sh > /root/ship_v41.log 2>&1 &
set -u
REPO=${REPO:-/root/ptcg/repo}
PENDING=${PENDING:-$REPO/data/sft/v41_pending}
I2BASE=${I2BASE:-/root/ptcg/repo/data/sft/v41_base_sft.jsonl.gz}
I2HOST=${I2HOST:-root@175.155.64.145}
I2PORT=${I2PORT:-19839}
I2KEY=${I2KEY:-/root/.ssh/id_i2}
IDLE_EXIT_MIN=${IDLE_EXIT_MIN:-0}      # 0 = run forever

say() { echo "[ship $(date -u +%m-%d_%H:%M:%S)] $*"; }
idle=0
say "watching $PENDING -> $I2HOST:$I2BASE"
while true; do
  F=$(ls -1 "$PENDING"/*.sft.jsonl.gz 2>/dev/null | head -1 || true)
  if [ -z "${F:-}" ]; then
    idle=$((idle + 1))
    if [ "$IDLE_EXIT_MIN" -gt 0 ] && [ "$idle" -ge "$IDLE_EXIT_MIN" ]; then
      say "nothing pending for ${idle} min -- exiting"; break
    fi
    sleep 60; continue
  fi
  idle=0
  TAG=$(basename "$F" .sft.jsonl.gz)
  WANT=$(zcat "$F" | wc -l)
  say "shipping $TAG ($WANT rows, $(du -h "$F" | cut -f1))"

  # scp wants -P for the port and ssh wants -p; spelled out rather than shared through a
  # variable because getting it wrong fails into a silent retry loop.
  if ! scp -q -i "$I2KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -P "$I2PORT" \
        "$F" "$I2HOST:/root/incoming_$TAG.gz"; then
    say "transfer of $TAG FAILED -- keeping it pending"; sleep 120; continue
  fi
  if ssh -i "$I2KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -p "$I2PORT" \
        "$I2HOST" "bash -s" <<REMOTE
set -u
B=$I2BASE; INC=/root/incoming_$TAG.gz
mkdir -p \$(dirname "\$B")
FREE=\$(df -Pk /root | awk 'NR==2 {print int(\$4/1048576)}')
[ "\$FREE" -lt 20 ] && { echo "[i2] only \$FREE GiB free -- refusing"; exit 1; }
GOT=\$(zcat "\$INC" | wc -l) || { echo "[i2] the batch does not decompress"; exit 1; }
[ "\$GOT" = "$WANT" ] || { echo "[i2] got \$GOT rows, expected $WANT -- transfer incomplete"; exit 1; }
if [ -s "\$B" ]; then cat "\$B" "\$INC" > "\$B.part" || exit 1; else cp "\$INC" "\$B.part" || exit 1; fi
python3 -c "
import gzip, json
n = 0
for line in gzip.open('\$B.part','rt'):
    json.loads(line); n += 1
print('[i2] pool reads clean: %d records' % n)
" || { rm -f "\$B.part"; exit 1; }
mv "\$B.part" "\$B"
rm -f "\$INC"
REMOTE
  then
    rm -f "$F"
    say "i2 pool extended with $TAG"
  else
    say "i2 append FAILED for $TAG -- keeping it pending"; sleep 120
  fi
done
