#!/usr/bin/env bash
# Add the two scouted decks' matchups to the v41 pools, at a share comparable to the decks
# already there. The existing 11,453,471 rows cover 1,953 pairs (~5,865 rows/pair); the new
# decks bring 127 pairs, so --games 48 targets ~745k rows rather than regenerating everything.
set -u
REPO=/root/ptcg/repo
BASE=$REPO/data/rerank/v41_base.jsonl.gz
TAG=newdecks_1
NEW=$REPO/data/rerank/$TAG.rerank.jsonl.gz
cd "$REPO"
say(){ echo "[newdecks $(date -u +%H:%M:%S)] $*"; }

say "generating (127 focused pairs x 48 games, 24 workers -- the GPU is training)"
CUDA_VISIBLE_DEVICES= nice -n 10 python3 tools/gen_selfplay.py --games 48 --workers 24 \
    --keep-blob --pair-with ogerpon_mono,dudunsparce_box --tag "$TAG" 2>&1 | tail -2 || exit 1
CUDA_VISIBLE_DEVICES= nice -n 10 python3 tools/build_rerank.py --tag "$TAG" --pfmt v41 \
    --label heuristic --sides both --workers 24 2>&1 | tail -2 || exit 1
[ -s "$NEW" ] || { say "build produced nothing"; exit 1; }

python3 - "$NEW" <<'PY' || { say "stamp/deck check FAILED -- discarding"; rm -f "$NEW"; exit 1; }
import gzip, json, sys, collections
n=bad=hits=0; decks=collections.Counter()
for line in gzip.open(sys.argv[1],"rt"):
    d=json.loads(line); n+=1
    if d.get("pfmt")!="v41": bad+=1
    decks[d.get("deck")]+=1
    s=d.get("state","")
    if any(k in s for k in (" d:"," d~"," n:"," thr"," tk:")): hits+=1
if bad or not n: raise SystemExit("pfmt missing on %d/%d"%(bad,n))
if hits*100 < n*5: raise SystemExit("only %d/%d rows carry a v41 fact -- --keep-blob?"%(hits,n))
new=sum(v for k,v in decks.items() if k in ("ogerpon_mono","dudunsparce_box"))
print("[check] %d rows, all v41, %d (%.1f%%) with a v41 fact | new-deck rows %d (%.1f%%)"
      %(n,hits,100.0*hits/n,new,100.0*new/n))
PY

cat "$BASE" "$NEW" > "$BASE.part" || exit 1
python3 - "$BASE.part" <<'PY' || { rm -f "$BASE.part"; exit 1; }
import gzip, json, sys
n=0
for line in gzip.open(sys.argv[1],"rt"): json.loads(line); n+=1
print("[check] pool reads clean: %d records"%n)
PY
mv "$BASE.part" "$BASE"
say "i1 pool now $(zcat "$BASE" | wc -l) rows"

rm -f /tmp/${TAG}_s*.gz
for i in 0 1 2 3 4 5 6 7; do
  nice -n 10 python3 tools/rerank_to_sft.py --inp "$NEW" --out /tmp/${TAG}_s$i.gz \
      --shard $i --nshards 8 > /dev/null 2>&1 &
done
wait
mkdir -p "$REPO/data/sft/v41_pending"
cat /tmp/${TAG}_s[0-7].gz > "$REPO/data/sft/v41_pending/$TAG.sft.jsonl.gz.part" \
  && mv "$REPO/data/sft/v41_pending/$TAG.sft.jsonl.gz.part" "$REPO/data/sft/v41_pending/$TAG.sft.jsonl.gz"
rm -f /tmp/${TAG}_s*.gz "$NEW"
rm -rf "$REPO/data/selfplay/$TAG"
say "queued for the shipper: $(ls -1 $REPO/data/sft/v41_pending | tr '\n' ' ')"
