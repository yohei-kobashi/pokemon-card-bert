#!/bin/bash
cd /root/ptcg/repo
python3 tools/dagger_to_sft.py --dagger data/rerank/dagger_i1.jsonl.gz \
  --base data/sft/v39_0731.jsonl.gz --ratio 0.05 \
  --out data/sft/v39_dag005.jsonl.gz
echo MIXDONE
