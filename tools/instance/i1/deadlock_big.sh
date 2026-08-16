#!/usr/bin/env bash
# Settle the deadlock rate. Two small runs disagreed (def 50% over 4 games through the staged
# BUNDLE, 3% over 8 games through the HF checkpoint), which is either sampling noise or a real
# artifact-vs-checkpoint difference -- INT8 ONNX on 2 CPU threads is not obviously the same
# pilot as fp32 on GPU. Both are measured here, at a size that can tell them apart.
set -u
cd /root/ptcg/repo
# 1) the ARTIFACT, on 2 pinned cores, the thing that actually ships
PYTHONPATH=cg-lib taskset -c 0-1 python3 tools/bench_bundle.py --stage /root/subm/dusk_v1 \
  --deck dragapult_dusknoir --opp alakazam_nz,marnie_grimmsnarl --games 20 \
  --out /root/deadlock_bundle.json > /root/deadlock_bundle.log 2>&1
echo "[bundle] rc=$?"
