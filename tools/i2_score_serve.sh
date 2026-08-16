#!/usr/bin/env bash
# instance2: keep the 4B score server alive on 127.0.0.1:$PORT.
#
# Binds to LOOPBACK on purpose. instance1 reaches it through the ssh tunnel that
# tools/i1_score_link.sh holds open, so the adapters are never exposed to the internet and
# there is no auth surface to get wrong. $PTCG_SCORE_TOKEN adds a second lock if set.
#
# WAITS FOR THE CARD. The server pins ~8 GiB for the base plus ~160 MiB of card-token rows, and
# a collection shard is 12-15 GiB: starting it under a running deck_lora2 would either OOM the
# shards or itself, and losing three shards costs a pass. So it waits until the GPU is quiet
# rather than racing it -- and it waits rather than exits, because a preflight that exits leaves
# nothing running and nothing complaining ([[two-instance-link-fails-silently]]).
set -u
cd /root/ptcg/repo
PORT=${PORT:-8077}
FREE_MIB=${FREE_MIB:-12000}       # headroom needed before starting
LOG=${LOG:-/root/score_server.log}
DECKS=${DECKS:-}                  # empty = every qwen: entry in the registry
say() { echo "[serve $(date -u +%m-%d_%H:%M:%S)] $*"; }

gpu_wait() {
    local waited=0
    while :; do
        local used total free
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        free=$((total - used))
        if [ "$free" -ge "$FREE_MIB" ] && ! pgrep -f 'deck_lora2\.sh|dpo_branch\.py' >/dev/null; then
            say "GPU free ${free} MiB, no training running -- starting"
            return 0
        fi
        [ $((waited % 600)) -eq 0 ] && say "waiting for the card (${free} MiB free, training $(pgrep -cf 'deck_lora2\.sh' || echo 0))"
        sleep 60; waited=$((waited + 60))
    done
}

while :; do
    gpu_wait
    say "launching score_server on 127.0.0.1:$PORT"
    PYTHONPATH=cg-lib:tools:. python3 -u tools/score_server.py \
        --from-registry ${DECKS:+--decks "$DECKS"} \
        --host 127.0.0.1 --port "$PORT" >> "$LOG" 2>&1
    say "score_server exited ($?) -- restarting in 30s"
    sleep 30
done
