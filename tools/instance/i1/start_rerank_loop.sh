#!/bin/bash
# Hand over from auto_eval_v40 to the loop: wait for its 63-deck screen to finish, then reuse
# that screen as round 1 instead of paying 1.7 h to repeat it.
cd /root/ptcg/repo
while pgrep -f "tools/auto_eval_v40" > /dev/null; do sleep 120; done
sleep 20
mkdir -p /root/loop_rerank
if [ -s /root/mirror_v40.json ]; then
  cp /root/mirror_v40.json /root/loop_rerank/mirror_r1.json
  export SKIP_FIRST_SCREEN=1
  echo "reusing the v40 screen as round 1" >> /root/loop_rerank/loop.log
fi
export KIND=rerank MODEL=/root/out/rerank_gte_v40 DEADLINE_H=24 SCREEN_GAMES=100 \
       COLLECT_GAMES=24 MAX_TARGETS=24
exec tools/dagger_loop2.sh
