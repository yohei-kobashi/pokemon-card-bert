#!/bin/bash
# Stop the reranker loop. Patterns are written with a bracket so they cannot match the ssh
# command line that carries them -- that mistake has killed the remote shell several times today.
for p in $(ps -eo pid,args | grep "[d]agger_loop2.sh" | awk '{print $1}'); do echo "kill loop $p"; kill $p; done
sleep 2
for p in $(ps -eo pid,args | grep "[t]rain_rerank.py" | awk '{print $1}'); do echo "kill train $p"; kill $p; done
sleep 8
echo "remaining: $(ps -eo args | grep -cE '[d]agger_loop2.sh|[t]rain_rerank.py')"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "--- loop state ---"
tail -3 /root/loop_rerank/loop.log
ls -la /root/loop_rerank/ /root/ptcg/repo/data/rerank/*.gz 2>/dev/null | awk '{print $5, $9}'
