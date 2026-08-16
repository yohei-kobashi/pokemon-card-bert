#!/bin/bash
# Restart from round 1 at a lower DAgger share, reusing v39's screen and collection.
cd /root/ptcg/repo
rm -rf /root/loop_rerank
mkdir -p /root/loop_rerank
cp /root/mirror_fleet.json /root/loop_rerank/mirror_r1.json
cp /root/ptcg/repo/data/rerank/dagger_v39a.jsonl.gz /root/loop_rerank/dagger_r1.jsonl.gz
export KIND=rerank MODEL=/root/out/rerank_gte_v39 DEADLINE_H=24 SCREEN_GAMES=100 \
       COLLECT_GAMES=24 MAX_TARGETS=24 RATIO=0.1 \
       SKIP_FIRST_SCREEN=1 SKIP_FIRST_COLLECT=1
exec tools/dagger_loop2.sh
