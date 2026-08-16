#!/usr/bin/env bash
# One-off: seed v41_base11 from the 42.5M-row pool, then hand off to the standing generator.
set -u
REPO=/root/ptcg/repo
B11=$REPO/data/rerank/v41_base11.jsonl.gz
say() { echo "[gen11 $(date -u +%m-%d_%H:%M:%S)] $*"; }
P11=$(PYTHONPATH=$REPO/tools python3 -c "import rl_config; print(\",\".join(rl_config.STAGE_C_TARGETS))")
say "extracting pilot-11 subset ($P11)"
python3 - "$REPO/data/rerank/v41_base.jsonl.gz" "$B11.part" "$P11" <<PY || { say "extract FAILED"; exit 1; }
import gzip, json, sys
src, dst, keep = sys.argv[1], sys.argv[2], set(sys.argv[3].split(","))
n = k = 0
with gzip.open(src, "rt") as f, gzip.open(dst, "wt") as g:
    for line in f:
        n += 1
        if json.loads(line).get("deck") in keep:
            g.write(line); k += 1
        if n % 5000000 == 0:
            print("[gen11]   %dM scanned, %d kept" % (n // 1000000, k), flush=True)
print("[gen11] extracted %d of %d rows (%.1f%%)" % (k, n, 100.0*k/max(1,n)), flush=True)
PY
mv "$B11.part" "$B11"
say "seeded $B11 ($(du -h "$B11" | cut -f1))"
# standing growth: pair-with-11 generation appending (filtered) to the 11-file
cd "$REPO"
BASE=$B11 TARGET_ROWS=30000000 setsid nohup bash tools/gen_pool_v41_resume.sh \
  >> /root/gen_v41_resume.log 2>&1 < /dev/null &
say "resume loop relaunched on $B11"
