#!/usr/bin/env bash
# instance1's branch DAEMON: serve pair-building requests from instance2.
#
# Why a daemon and not a push pipeline: instance2 cannot reach instance1 at all (the vast ssh
# proxy authenticates against the account's keys, so an authorized_keys entry on instance1 is
# never consulted), while instance1 CAN reach instance2 -- so instance1 must poll. The i1->i2
# key is kept alive by keyheal.sh on instance2, which re-appends it within a minute of vast's
# periodic authorized_keys rewrite (both failure directions were measured today, 2026-08-10).
#
# Protocol, all files on INSTANCE2:
#   /root/branch_request        written by round.sh: a single line, the round number
#   /root/traces_rN.s*.jsonl.gz the collection shards for that round
#   /root/dpo_rN.jsonl.gz       the reply; round.sh waits for it
# The request is removed ONLY after the reply has landed, so a crash at any point leaves the
# request in place and the next poll retries the whole job idempotently.
set -u
I2="-i /root/.ssh/id_i2 -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=20"
I2HOST=root@175.155.64.145
I2PORT=19839
REPO=/root/ptcg/repo
say() { echo "[branchd $(date -u +%m-%d_%H:%M:%S)] $*"; }

say "serving (poll every 120s)"
while :; do
    N=$(ssh $I2 -p $I2PORT $I2HOST 'cat /root/branch_request 2>/dev/null' 2>/dev/null | tr -dc 0-9)
    if [ -z "$N" ]; then
        sleep 120
        continue
    fi
    say "request: round $N"
    rm -f /root/traces_r$N.s*.jsonl.gz
    if ! scp $I2 -P $I2PORT "$I2HOST:/root/traces_r$N.s*.jsonl.gz" /root/ 2>/dev/null; then
        say "pull failed (link down?); retrying next poll"
        sleep 120
        continue
    fi
    cd "$REPO"
    # 32 workers, not 40: the GPU gates running alongside also eat CPU for their engine
    # opponents, and the quota memory is explicit about never oversubscribing.
    if CUDA_VISIBLE_DEVICES= PYTHONPATH=cg-lib nice -n 5 python3 tools/dpo_branch.py \
        --traces "$(ls /root/traces_r$N.s*.jsonl.gz | paste -sd,)" \
        --budget 20000 --per-game 15 --margin-min 0.01 --playouts 32 --workers 32 \
        --out /root/dpo_r$N.jsonl.gz > /root/branch_r$N.log 2>&1; then
        NP=$(zcat /root/dpo_r$N.jsonl.gz | wc -l)
        if scp $I2 -P $I2PORT /root/dpo_r$N.jsonl.gz "$I2HOST:/root/dpo_r$N.jsonl.gz"; then
            ssh $I2 -p $I2PORT $I2HOST 'rm -f /root/branch_request'
            say "round $N served: $NP pairs"
        else
            say "SHIP FAILED for round $N; will retry next poll"
        fi
    else
        say "branch FAILED for round $N (see /root/branch_r$N.log); clearing request"
        tail -5 /root/branch_r$N.log
        # Clear the request so round.sh's timeout fires its local fallback instead of the
        # daemon retrying a deterministic failure forever.
        ssh $I2 -p $I2PORT $I2HOST 'rm -f /root/branch_request'
    fi
    sleep 30
done
