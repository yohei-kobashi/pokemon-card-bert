#!/usr/bin/env bash
set -u
cd /root/ptcg/repo
for D in crustle_geco crustle_stall; do
  for O in ogerpon_mono slowking dudunsparce_box; do
    nohup python3 /root/baseline_one.py "$D" "$O" 300 /root/out/fleet_baseline/${D}__${O}.json > /root/out/fleet_baseline/${D}__${O}.log 2>&1 &
  done
done
wait
echo CELLS_DONE
