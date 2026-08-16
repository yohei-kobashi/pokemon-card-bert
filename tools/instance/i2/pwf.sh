#!/usr/bin/env bash
# Run the (deck, kind) pricing as soon as the GPU has room for the 4B.
#
# The 4B needs ~12 GiB and instance2's screen holds 3 shards x 12 GiB of a 47 GiB card, so a
# launch during a screen dies while resizing the embedding table -- observed 2026-08-06 11:12,
# "Tried to allocate 1.45 GiB ... 1.32 GiB is free". Polling costs nothing and loses no time
# whichever way the schedule falls.
#
#   nohup bash tools/price_when_free.sh > /root/price_watch.log 2>&1 &
set -u
REPO=${REPO:-/root/ptcg/repo}
NEED=${NEED:-14000}                       # MiB free before trying (12 GiB model + headroom)
TARGETS=${TARGETS:-/root/lm_targets_i2r6.json}
MODEL=${MODEL:-qwen:/root/out/i2_r6}
OUT=${OUT:-/root/priced_engine.json}
LOG=${LOG:-/root/price_engine.log}
TOP=${TOP:-6}
POINTS=${POINTS:-100}
MAX_TRIES=${MAX_TRIES:-3}

say() { echo "[price $(date -u +%m-%d_%H:%M:%S)] $*"; }
cd "$REPO" || { say "cannot cd $REPO"; exit 1; }

tries=0
while true; do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  case "${FREE:-}" in
    ''|*[!0-9]*) say "nvidia-smi gave no number -- retrying in 5 min"; sleep 300; continue ;;
  esac
  if [ "$FREE" -lt "$NEED" ]; then
    say "only ${FREE} MiB free, need ${NEED} -- waiting"
    sleep 300
    continue
  fi
  tries=$((tries + 1))
  say "GPU has ${FREE} MiB free -- starting pricing (try $tries)"
  PYTHONPATH=cg-lib python3 tools/price_targets.py \
      --targets "$TARGETS" --model "$MODEL" --rollout engine \
      --points "$POINTS" --games 300 --playouts 16 --top "$TOP" \
      --out "$OUT" >> "$LOG" 2>&1
  rc=$?
  say "pricing exited rc=$rc"
  [ $rc -eq 0 ] && break
  if [ "$tries" -ge "$MAX_TRIES" ]; then
    say "giving up after $tries tries -- see $LOG"
    exit 1
  fi
  sleep 600
done
say "done -> $OUT"
