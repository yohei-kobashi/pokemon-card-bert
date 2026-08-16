#!/bin/bash
set -u
cd /root/ptcg/repo
cp /tmp/dedup_rerank.py /tmp/build_rerank.py /tmp/collect_dagger.py tools/
cp /tmp/action_token.py lm/
echo "=== base pool ==="
python3 tools/dedup_rerank.py --inp data/rerank/v39_0731.rerank.jsonl.gz \
                              --out data/rerank/v39_0731.dd.rerank.jsonl.gz
for r in 1 2; do
  echo "=== dagger r$r ==="
  python3 tools/dedup_rerank.py --inp /root/loop_rerank/dagger_r$r.jsonl.gz \
                                --out /root/loop_rerank/dagger_r$r.dd.jsonl.gz
done
echo REBUILD DONE
