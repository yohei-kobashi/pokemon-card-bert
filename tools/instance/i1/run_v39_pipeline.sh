set -u
cd /root/ptcg/repo
export PYTHONPATH=/root/ptcg/repo:/root/ptcg/repo/cg-lib:/root/ptcg/repo/tools
TAG=v39_0731
say () { echo "[$(date -u +%H:%M:%S)] $*"; }

say "=== 1/4 gen_selfplay (current engine) ==="
CUDA_VISIBLE_DEVICES= python3 tools/gen_selfplay.py --games 12 --workers 96 --tag $TAG 2>&1 | tail -3

say "=== 2/4 build_rerank (pfmt=current => v39) ==="
CUDA_VISIBLE_DEVICES= python3 tools/build_rerank.py --tag $TAG --glossary none \
  --deck-mode roles --label heuristic --sides both --pfmt current 2>&1 | tail -6

F=$(ls -t data/rerank/${TAG}*.jsonl.gz 2>/dev/null | head -1)
say "built: $F"
[ -z "$F" ] && { say "NO OUTPUT -- stopping"; exit 1; }

say "=== 3/4 verify the data actually carries v39 ==="
CUDA_VISIBLE_DEVICES= python3 - "$F" <<'PY'
import gzip, json, sys
f = sys.argv[1]
n = has_roles = has_need = has_rt = has_idme = 0
first = None
for line in gzip.open(f, "rt"):
    d = json.loads(line); n += 1
    s = d.get("state") or d.get("prompt") or ""
    if "DECK win[" in s or "DECK eng[" in s or "DECK line[" in s: has_roles += 1
    if "need:" in s: has_need += 1
    if "rt:" in s: has_rt += 1
    if "ID ME" in s: has_idme += 1
    if first is None: first = s
    if n >= 40000: break
print("  records sampled %d" % n)
print("  DECK role groups : %5.1f%%" % (100.0*has_roles/n))
print("  need:            : %5.1f%%" % (100.0*has_need/n))
print("  rt:              : %5.1f%%" % (100.0*has_rt/n))
print("  ID ME (must be 0): %5.1f%%" % (100.0*has_idme/n))
print("  sample:", first[:220])
bad = (has_roles < 0.9*n) or (has_rt < 0.5*n) or (has_idme > 0)
sys.exit(2 if bad else 0)
PY
[ $? -ne 0 ] && { say "VERIFICATION FAILED -- not starting training"; exit 1; }

say "=== 4/4 train_rerank from scratch on v39 ==="
nohup setsid python3 tools/train_rerank.py --data "$F" \
  --out /root/out/rerank_gte_v39 --deadline-h 6 --max-samples 800000 \
  --pair-batch 48 --accum 8 --lr 2e-5 --max-len 640 --eval-n 2000 \
  > /root/train_v39.log 2>&1 < /dev/null &
sleep 20
say "training launched, log /root/train_v39.log"
head -12 /root/train_v39.log 2>/dev/null
echo PIPELINE_STARTED
