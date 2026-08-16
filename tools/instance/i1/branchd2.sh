#!/usr/bin/env bash
# instance1's SECOND branch daemon: tag-based, for the per-deck opponent adapters.
#
# branchd.sh is round-numbered and hard-wires /root/traces_rN / /root/dpo_rN; the per-deck loop
# runs five decks concurrently in time and needs its own namespace. Same direction of travel and
# the same reason (instance2 cannot reach instance1 -- the vast proxy authenticates on account
# keys -- so instance1 polls).
#
# Protocol, all files on INSTANCE2:
#   /root/branch_request2         TAG|ONLY_DECK|BUDGET|PLAYOUTS   (one line)
#   /root/traces_<TAG>.s*.jsonl.gz  the collection shards
#   /root/pairs_<TAG>.jsonl.gz      the reply; deck_lora.sh waits for it
# The request is cleared ONLY after the reply lands, so any crash retries the whole job.
#
# WORKERS: 24, not 32. branchd.sh may still be serving the last fleet round and instance1 has
# 61.4 EFFECTIVE cores ([[vast-cpu-quotas]] -- nproc reports 112 and lies); 32+32 oversubscribes.
set -u
I2="-i /root/.ssh/id_i2 -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=20"
I2HOST=root@175.155.64.145
I2PORT=19839
REPO=/root/ptcg/repo
say() { echo "[branchd2 $(date -u +%m-%d_%H:%M:%S)] $*"; }

say "serving (poll every 120s)"
while :; do
    REQ=$(ssh $I2 -p $I2PORT $I2HOST 'touch /root/.branchd2_alive; cat /root/branch_request2 2>/dev/null' 2>/dev/null \
          | tr -dc 'A-Za-z0-9_|.-')
    if [ -z "$REQ" ]; then
        sleep 120
        continue
    fi
    TAG=$(echo "$REQ" | cut -d'|' -f1)
    ONLY=$(echo "$REQ" | cut -d'|' -f2)
    BUDGET=$(echo "$REQ" | cut -d'|' -f3)
    PLAYOUTS=$(echo "$REQ" | cut -d'|' -f4)
    PERGAME=$(echo "$REQ" | cut -d'|' -f5)
    [ -n "$TAG" ] && [ -n "$ONLY" ] || { say "malformed request %$REQ% -- clearing"; \
        ssh $I2 -p $I2PORT $I2HOST 'rm -f /root/branch_request2'; continue; }
    BUDGET=${BUDGET:-12000}
    PLAYOUTS=${PLAYOUTS:-24}
    PERGAME=${PERGAME:-15}        # 5th field is optional; old requests still work
    say "request: tag=$TAG deck=$ONLY budget=$BUDGET playouts=$PLAYOUTS per-game=$PERGAME"

    rm -f /root/traces_$TAG.s*.jsonl.gz
    if ! scp $I2 -P $I2PORT "$I2HOST:/root/traces_$TAG.s*.jsonl.gz" /root/ 2>/dev/null; then
        say "could not fetch traces for $TAG -- will retry"
        sleep 120
        continue
    fi
    TR=$(ls /root/traces_$TAG.s*.jsonl.gz 2>/dev/null | paste -sd,)
    [ -n "$TR" ] || { say "no traces after fetch -- retrying"; sleep 120; continue; }

    cd $REPO
    # Prize-shaped terminal reward, matching the mirror chain: the +-1 outcome plus a margin
    # term. No --rule-weights here -- the plan rules are dusknoir's, and these adapters pilot
    # the OPPONENT.
    RL_PRIZE_GAMMA=0.25 CUDA_VISIBLE_DEVICES= PYTHONPATH=cg-lib nice -n 5 \
        python3 tools/dpo_branch.py --traces "$TR" --only-deck "$ONLY" \
        --budget "$BUDGET" --per-game "$PERGAME" --margin-min 0.01 \
        --playouts "$PLAYOUTS" --workers 24 --seed 41000 \
        --out /root/pairs_$TAG.jsonl.gz > /root/branch_$TAG.log 2>&1
    if [ ! -s /root/pairs_$TAG.jsonl.gz ]; then
        say "branch FAILED for $TAG:"; tail -6 /root/branch_$TAG.log
        ssh $I2 -p $I2PORT $I2HOST 'rm -f /root/branch_request2'
        continue
    fi
    NP=$(zcat /root/pairs_$TAG.jsonl.gz | wc -l)
    if scp $I2 -P $I2PORT /root/pairs_$TAG.jsonl.gz "$I2HOST:/root/pairs_$TAG.jsonl.gz.part" \
       && ssh $I2 -p $I2PORT $I2HOST "gzip -t /root/pairs_$TAG.jsonl.gz.part \
              && mv -f /root/pairs_$TAG.jsonl.gz.part /root/pairs_$TAG.jsonl.gz"; then
        ssh $I2 -p $I2PORT $I2HOST 'rm -f /root/branch_request2'
        say "$TAG served: $NP pairs"
    else
        say "$TAG: $NP pairs built but the ship FAILED -- request left in place"
    fi
done
