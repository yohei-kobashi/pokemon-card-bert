#!/bin/bash
# Grow the base pool with self-play from TODAY's engine, then swap it in under the running loop.
#
# WHY NOT THE LOGS ALREADY ON DISK. `curengine_0724` and `v34_full` are unused and would cost no
# generation at all, but they are from 24 and 21 July and the engine has had many fixes since
# (dead-card rules, the spare-ex bench guard, the honchkrow profile, mega_lucario). Imitating an
# older pilot is a worse teacher, and the base pool is 85-90% of every round.
#
# 28 WORKERS, not the 96 the old pipeline used. The box's cgroup quota is 61.4 cores (nproc says
# 256 and is wrong), and dagger_loop3 needs CPU for its own screening and collection. Half each
# means both run at about half speed, which is the price of not having to sequence them.
#
# THE SWAP IS AN mv, which is atomic within a filesystem: a mix already reading the old file
# keeps its inode and finishes on the old data, and the next round opens the new one. No round
# ever sees a half-written pool.
set -u
REPO=/root/ptcg/repo
TAG=cur_0802
BASE=$REPO/data/rerank/v40_base.jsonl.gz
LOG=/root/grow_base.log
cd "$REPO"
exec >> "$LOG" 2>&1
say() { echo "[grow $(date -u +%m-%d_%H:%M:%S)] $*"; }

FREE=$(df -Pk /root | awk 'NR==2 {print int($4/1048576)}')
say "starting | $FREE GiB free | tag $TAG"
[ "$FREE" -lt 12 ] && { say "STOP: under 12 GiB free, refusing to generate"; exit 1; }

say "=== 1/3 self-play with the current engine (28 workers) ==="
CUDA_VISIBLE_DEVICES= nice -n 15 python3 tools/gen_selfplay.py --games 12 --workers 28 \
    --tag "$TAG" 2>&1 | tail -4 || { say "STOP: generation failed"; exit 1; }

FREE=$(df -Pk /root | awk 'NR==2 {print int($4/1048576)}')
say "generation done | $FREE GiB free"
[ "$FREE" -lt 8 ] && { say "STOP: under 8 GiB free after generating, not building"; exit 1; }

say "=== 2/3 build rerank records in the CURRENT prompt format ==="
CUDA_VISIBLE_DEVICES= nice -n 15 python3 tools/build_rerank.py --tag "$TAG" --pfmt current \
    --label heuristic --sides both --workers 28 2>&1 | tail -5 \
    || { say "STOP: build failed"; exit 1; }
NEW=$(ls -t data/rerank/${TAG}*.rerank.jsonl.gz 2>/dev/null | head -1)
[ -s "$NEW" ] || { say "STOP: the build produced no file"; exit 1; }
say "built $NEW"

# The new records must be in the SAME format as the pool they join. Building with --pfmt current
# should guarantee it, but a mixed-format pool trains a model on inputs it will never see and
# says nothing on the way past, so check rather than trust.
python3 - "$NEW" "$BASE" <<'PY' || { say "STOP: the new records do not match the pool's format"; exit 1; }
import gzip, json, sys
sys.path.insert(0, "."); sys.path.insert(0, "cg-lib"); sys.path.insert(0, "tools")
from menu_dedup_pool import rewrite
def profile(path, cap=3000):
    n = roles = dedup = facts = 0
    for line in gzip.open(path, "rt"):
        s = json.loads(line).get("state") or ""
        if " :: " not in s:
            continue
        n += 1
        roles += ("DECK win[" in s or "DECK eng[" in s or "DECK line[" in s)
        facts += ("need:" in s)
        dedup += (rewrite(s)[0] == s)
        if n >= cap:
            break
    return n, roles / n, facts / n, dedup / n
a = profile(sys.argv[1]); b = profile(sys.argv[2])
print("[fmt] new  n=%d roles=%.2f need=%.2f menu-deduped=%.2f" % a)
print("[fmt] pool n=%d roles=%.2f need=%.2f menu-deduped=%.2f" % b)
ok = all(abs(x - y) < 0.05 for x, y in zip(a[1:], b[1:]))
raise SystemExit(0 if ok else 1)
PY

say "=== 3/3 extend the base pool and swap it in ==="
cat "$BASE" "$NEW" > "$REPO/data/rerank/v40_base_ext.jsonl.gz.part" \
    || { say "STOP: concatenation failed"; exit 1; }
python3 - "$REPO/data/rerank/v40_base_ext.jsonl.gz.part" <<'PY' || { say "STOP: the extended pool is not readable end to end"; exit 1; }
import gzip, json, sys
n = 0
for line in gzip.open(sys.argv[1], "rt"):
    json.loads(line)
    n += 1
print("[check] extended pool reads clean: %d records" % n)
PY
mv "$REPO/data/rerank/v40_base_ext.jsonl.gz.part" "$BASE"
say "SWAPPED: $BASE is now $(zcat $BASE | wc -l) records; later rounds pick it up automatically"
