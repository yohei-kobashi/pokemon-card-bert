set -u
REPO=/root/ptcg/repo; cd $REPO
B="$1"; FIELD="$2"
echo "[patch] target $B field=$FIELD"
BEFORE=$(zcat "$B" | wc -l) || exit 1
echo "[patch] before: $BEFORE rows"
PYTHONPATH=cg-lib python3 tools/patch_rt.py --inp "$B" --out "$B.part" --field "$FIELD" || exit 1
AFTER=$(zcat "$B.part" | wc -l) || { echo "[patch] .part does not decompress"; rm -f "$B.part"; exit 1; }
[ "$AFTER" = "$BEFORE" ] || { echo "[patch] ROW COUNT CHANGED $BEFORE -> $AFTER, refusing"; rm -f "$B.part"; exit 1; }
python3 -c "
import gzip, json, sys
n=0
for line in gzip.open('$B.part','rt'):
    json.loads(line); n+=1
print('[patch] reads clean: %d records' % n)
" || { rm -f "$B.part"; exit 1; }
mv "$B.part" "$B"
echo "[patch] swapped in"
