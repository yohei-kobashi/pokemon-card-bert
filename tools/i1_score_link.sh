#!/usr/bin/env bash
# instance1: hold open the tunnel to instance2's score server, and SAY when it is not there.
#
#   http://127.0.0.1:8077  on instance1  ==  the 4B server on instance2
#
# so a spec reads `remote:http://127.0.0.1:8077|lora_marnie_grimmsnarl_r1` and nothing in the
# gate has to know another machine exists.
#
# WHY A SUPERVISOR AND NOT JUST `ssh -L`. The link between these two boxes has failed silently
# before: vast rewrites authorized_keys on restart, and a one-shot tunnel dies without anything
# downstream saying so. A dropped tunnel does NOT stop a gate -- lm/agent.py catches every
# scorer error and falls back to engine_v2 -- so the run would keep going and quietly measure
# the wrong opponent. lm/remote_scorer.py kills a worker after 8 consecutive failures; this
# script is the other half, keeping the outage short enough that it never gets there.
set -u
PORT=${PORT:-8077}
I2H=${I2H:-root@175.155.64.145}
I2P=${I2P:-19839}
KEY=${KEY:-/root/.ssh/id_i2}
say() { echo "[link $(date -u +%m-%d_%H:%M:%S)] $*"; }

healthy() { curl -sf -m 10 "http://127.0.0.1:$PORT/health" 2>/dev/null; }

down=0
while :; do
    if H=$(healthy); then
        [ "$down" -gt 0 ] && say "back up after ${down}s: $H"
        down=0
        sleep 30
        continue
    fi
    # The tunnel may be up while the server is down (or still loading its 8 GiB base): tearing
    # it down and redialling is harmless and fixes the case where the tunnel is the broken half.
    pkill -f "ssh -N -L $PORT:127.0.0.1:$PORT" 2>/dev/null
    sleep 1
    ssh -N -L "$PORT:127.0.0.1:$PORT" -i "$KEY" -p "$I2P" \
        -o StrictHostKeyChecking=no -o ExitOnForwardFailure=yes \
        -o ServerAliveInterval=15 -o ServerAliveCountMax=3 "$I2H" &
    sleep 10
    if healthy >/dev/null; then
        say "tunnel up"
        down=0
    else
        [ $((down % 300)) -eq 0 ] && say "no score server behind the tunnel (${down}s) -- is tools/i2_score_serve.sh running on instance2?"
        down=$((down + 15))
        sleep 5
    fi
done
