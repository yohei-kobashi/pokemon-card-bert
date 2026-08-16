#!/usr/bin/env bash
# The full 9-cell protocol at 300 games/cell for v35 and v36 -- the SAME sample size as the
# engine_v2 baseline in /root/out/base_grid300, so every LM-vs-engine delta is finally read
# at matched precision (SE 1.7pt per deck instead of 3.7pt at 60 games).
#
# We ran 60 games/cell for weeks believing a 9-cell run cost hours. It does on the ONNX CPU
# deploy path; on torch/CUDA 450 games took 4 minutes. The cheap measurement was available
# the whole time -- check the cost before designing around it.
#
# Each model in ITS OWN prompt format: v35 static/no-shuffle, v36 remaining/shuffle.
set -u
cd /root/ptcg/repo
export PYTHONPATH=cg-lib
while pgrep -f "tiebrea[k]_cs" > /dev/null; do sleep 20; done
echo "=== hiN start $(date -u) ==="
tools/eval_rerank_par.sh /root/out/wr_v35_300 /root/out/rerank_gte_v35 torch "" 8 300 1000000 none static 0
echo "=== v35 300G done $(date -u) ==="
tools/eval_rerank_par.sh /root/out/wr_v36_300 /root/out/rerank_gte_v36 torch "" 8 300 1000000 none remaining 1
echo "=== HIN_ALL_DONE $(date -u) ==="
