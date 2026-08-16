#!/usr/bin/env bash
# instance1 side: collect the traces instance2 is generating, because instance2 cannot push.
#
# The vast proxy authenticates on account keys, so the i2 -> i1 direction is closed and always
# has been; branchd2 exists in the same shape for the same reason. This polls, fetches whole
# shards, verifies the gzip before accepting, and deletes the far copy only once the near copy
# is proven good.
set -u
LOG=/root/genpull.log
I2="-i /root/.ssh/id_i2 -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=20"
H=root@175.155.64.145; P=19839
IN=/root/gen_in
say() { echo "[pull $(date -u +%m-%d_%H:%M:%S)] $*" >> "$LOG"; }
mkdir -p "$IN"
say "polling instance2 for generated traces every 120s"

while :; do
    LIST=$(ssh $I2 -p $P $H 'ls -1 /root/gen_out/gtr_*.jsonl.gz 2>/dev/null' 2>/dev/null)
    if [ -z "$LIST" ]; then sleep 120; continue; fi
    for F in $LIST; do
        B=$(basename "$F")
        [ -s "$IN/$B" ] && continue                      # already have it
        if scp $I2 -P $P -q "$H:$F" "$IN/$B.part" 2>/dev/null && gzip -t "$IN/$B.part" 2>/dev/null; then
            mv -f "$IN/$B.part" "$IN/$B"
            N=$(zcat "$IN/$B" | wc -l)
            say "took $B ($N games)"
            ssh $I2 -p $P $H "rm -f $F" 2>/dev/null       # only after it verified here
        else
            rm -f "$IN/$B.part"
            say "fetch of $B failed -- will retry"
        fi
    done
    sleep 120
done
