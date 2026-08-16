#!/bin/bash
# Wait for instance1's base pool to finish copying, VERIFY it, then start instance2's DAgger loop.
#
# The verification is not ceremony. A truncated .gz decompresses fine up to the cut, so a partial
# transfer would train on a partial pool and nothing would error -- the same silent-wrong-data
# failure mode that has cost this project several runs. The row count must be exactly what
# instance1 wrote (5,733,620) before the file is allowed to become the base.
set -u
LOG=/root/start_i2.log
exec >> "$LOG" 2>&1
say() { echo "[start_i2 $(date -u +%m-%d_%H:%M:%S)] $*"; }

DST=/root/ptcg/repo/data/sft/v40_base_sft.jsonl.gz
PART=$DST.part
WANT=5733620

say "waiting for $PART to stop growing"
last=0
for _ in $(seq 1 900); do          # up to 7.5 hours: the link fell from 240 KB/s to 46 KB/s mid-copy
  [ -s "$PART" ] || { sleep 30; continue; }
  cur=$(stat -c %s "$PART")
  if [ "$cur" = "$last" ] && [ "$cur" -gt 400000000 ]; then break; fi
  last=$cur
  sleep 30
done
say "size settled at $(stat -c %s "$PART") bytes"

n=$(zcat "$PART" | wc -l) || { say "STOP: the file does not decompress"; exit 1; }
if [ "$n" != "$WANT" ]; then say "STOP: $n rows, expected $WANT -- transfer is incomplete"; exit 1; fi
say "verified $n rows"
mv "$PART" "$DST"

cd /root/ptcg/repo
MODEL=/root/out/qwen3_4b_cfb_v40 DEADLINE_H=96 \
  nohup bash tools/dagger_loop_i2.sh > /dev/null 2>&1 &
sleep 10
say "loop started (pid $(pgrep -f '[d]agger_loop_i2.sh' | head -1))"
