#!/bin/bash
# Diagnose every deck the mirror screen marked WORSE.
#
# Waits for the screen to finish rather than reading a partial file: mirror_match writes its
# JSON only at the end, and diagnosing a half-finished list would quietly skip the decks that
# had not been reached yet.
set -u
REPO=/root/ptcg/repo
JSON=${1:-/root/mirror_fleet.json}
MODEL=${2:-hf:/root/out/rerank_gte_v39}
GAMES=${3:-16}
OUT=/root/diag_worse
mkdir -p "$OUT"
cd "$REPO"

while pgrep -f "tools/mirror_match" > /dev/null; do sleep 60; done
sleep 5
[ -f "$JSON" ] || { echo "no $JSON -- the screen did not finish"; exit 1; }

DECKS=$(python3 -c "
import json,sys
d=json.load(open('$JSON'))['decks']
print(' '.join(k for k,v in d.items() if v['verdict']=='WORSE'))")
echo "WORSE decks: $DECKS"
echo "$DECKS" > "$OUT/list.txt"

for d in $DECKS; do
  echo "=== $d ==="
  PYTHONPATH=cg-lib python3 tools/diag_pilot.py --deck "$d" --model "$MODEL" \
      --games "$GAMES" --out "$OUT/$d.json" 2>&1 | grep -vE "Loading weights|^\s*$"
done
echo "ALL DONE"
