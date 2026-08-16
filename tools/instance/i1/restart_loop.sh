#!/bin/bash
# Restart the reranker loop on the deduped pools.
#
# Round 1 reuses the existing screen and collection deliberately: the screen measures the v39
# model, which has not changed, and the collection has been rewritten in place by
# tools/dedup_rerank.py rather than regenerated. What DOES change is the training data, which is
# the whole reason the loop was stopped -- so round 1 trains from base on clean pools.
set -u
mv -f /root/loop_rerank/loop.log /root/loop_rerank/loop.predup.log 2>/dev/null
cd /root/ptcg/repo
KIND=rerank \
MODEL=/root/out/rerank_gte_v39 \
RATIO=0.1 \
DEADLINE_H=24 \
SKIP_FIRST_SCREEN=1 \
SKIP_FIRST_COLLECT=1 \
setsid nohup /root/ptcg/repo/tools/dagger_loop2.sh > /dev/null 2>&1 < /dev/null &
sleep 45
echo "loop procs: $(ps -eo args | grep -c '[d]agger_loop2.sh')"
tail -20 /root/loop_rerank/loop.log
