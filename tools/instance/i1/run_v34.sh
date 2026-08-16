#!/bin/bash
cd ~/ptcg/repo
chmod 755 cg-lib/cg/*.so 2>/dev/null
export PYTHONPATH="$PWD:$PWD/cg-lib"
rm -rf data/selfplay/v34_full data/sft
echo "=== [1/2] self-play: 250 workers, 63 decks x 40 games ==="
date
time python tools/gen_selfplay.py --games 40 --tag v34_full --lean --workers 250
echo "=== [2/2] build_sft: 48 workers (value scorer + ID-OP) ==="
date
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python tools/build_sft.py \
  --tag v34_full --modes act --value-model out/value/value_data.npz \
  --turn-boundary 2.75 --eval-margin 4.0 --eval-temp 1.5 --workers 48
echo "=== DONE ==="; date; ls -la data/sft/
