#!/usr/bin/env bash
# Wait for the bundles, then run each one the way Kaggle will.
set -u
export PYTHONPATH=/root/ptcg/repo/cg-lib HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
say() { echo "[smoke $(date -u +%m-%d_%H:%M:%S)] $*"; }
while ! grep -q "BUILD_BUNDLES_DONE" /root/build_bundles.log 2>/dev/null; do sleep 20; done

for TAG in dusk_s1_pure dusk_s1_attach; do
    [ -d "/root/subm/$TAG" ] || { say "$TAG was never staged -- build failed"; continue; }
    say "================ $TAG ================"
    python3 /root/smoke_bundle.py "/root/subm/$TAG" --games 4 --opp crustle 2>&1 | tail -25
done

# The fallback path, on the bundle we would actually ship. A 3-second budget is spent inside the
# first decision or two, so almost the whole game is played by engine_v2 through the LM adapter:
# the question is not whether that plays well, it is whether the raise is CAUGHT. An uncaught
# one forfeits the game, and a forfeit on the ladder is indistinguishable from a bad agent.
say "================ time-bank fallback (budget 3s) ================"
python3 /root/smoke_bundle.py /root/subm/dusk_s1_attach --games 2 --opp crustle \
    --tiny-budget 3 2>&1 | tail -15
say "SMOKE_ALL_DONE"
