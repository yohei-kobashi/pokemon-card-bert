#!/bin/bash
cd /root/ptcg/repo
export KIND=qwen MODEL=/root/out/teacher9b_v39 DEADLINE_H=24 SCREEN_GAMES=60 \
       COLLECT_GAMES=24 MAX_TARGETS=20 SFT_LIMIT=100000
exec tools/dagger_loop.sh
