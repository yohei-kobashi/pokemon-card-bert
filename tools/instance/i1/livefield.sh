#!/usr/bin/env bash
# The submission decision measured against the LIVE FIELD, not the historical 3-opponent set.
#
# The 9-cell protocol faces alakazam / crustle / dragapult = 0.086 + 0.048 + 0.044 = 17.8% of
# the top-500 ladder (tools/rl_config.LIVE_META, scouted 2026-07-23), and OMITS the two most
# common opponents entirely: alakazam_nz (0.212, #1) and marnie_grimmsnarl (0.172, #2). Worse,
# the single opponent where both LMs trail engine_v2 is alakazam -- a stand-in for a 34.4%
# alakazam family -- while the two they beat are 9.2% of the field combined. A protocol that
# over-weights our strengths and omits half the ladder cannot decide a submission.
#
# 7 opponents = 70.4% of the live field. engine_v2 baseline runs on the SAME grid (CPU) so
# every LM number has its own control at the same sample size.
set -u
cd /root/ptcg/repo
export PYTHONPATH=cg-lib
OPPS_LIST="alakazam_nz marnie_grimmsnarl archaludon alakazam cynthia_garchomp crustle dragapult"
while pgrep -f "hi[N].sh" > /dev/null; do sleep 20; done
echo "=== livefield start $(date -u) ==="

mkdir -p /root/out/base_live
for o in $OPPS_LIST; do
  [ -f /root/out/base_live/$o.json ] && continue
  nice -n 5 nohup python3 /root/baseline_one.py crustle_stall $o 300 /root/out/base_live/$o.json \
    > /root/out/base_live/$o.log 2>&1 &
done

for M in v35 v36; do
  if [ "$M" = v35 ]; then MODE=static; SH=0; else MODE=remaining; SH=1; fi
  echo "=== live field rerank_gte_$M ($(date -u)) ==="
  DECKS="crustle_stall" OPPS="$OPPS_LIST" \
    tools/eval_rerank_par.sh "/root/out/wr_${M}_live" "/root/out/rerank_gte_$M" torch "" 8 300 1000000 none $MODE $SH
done
wait
echo "=== LIVEFIELD_ALL_DONE $(date -u) ==="
