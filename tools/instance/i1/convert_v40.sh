#!/bin/bash
set -u
cd /root/ptcg/repo
exec >> /root/convert_v40.log 2>&1
for f in "data/rerank/v39_0731.rerank.jsonl.gz v40_base" "/root/loop_rerank/dagger_r1.jsonl.gz v40_dagger_r1" "data/rerank/attach_q1c.jsonl.gz v40_attach_q1" "data/rerank/attach_qheld.jsonl.gz v40_attach_held"; do
  set -- $f
  echo "=== $2 ==="
  nice -n 12 python3 tools/menu_dedup_pool.py --inp "$1" --out "data/rerank/$2.jsonl.gz"
done
echo "CONVERSIONS DONE"
