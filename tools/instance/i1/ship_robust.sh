#!/usr/bin/env bash
# Move the checkpoint with a per-chunk timeout, because on this link some connections stall.
#
# OBSERVED TWICE, IDENTICALLY: of six parallel scp streams, most finish at 1-11 MB/s and two or
# three settle at 40-90 kB/s and stay there. They do not recover when the fast ones finish, and
# killing a stalled one and re-opening it immediately gets full speed -- so it is the individual
# connection that is in a bad state, not the path and not the far host. I could not identify the
# mechanism; what is repeatable is the remedy.
#
# So: give every chunk a deadline, and on expiry kill it and try again. 62 MB at the good rate is
# ~6-60 s, so a 150 s cap only ever fires on a stream that has already gone bad.
set -u
N=${N:-12}
TMO=${TMO:-150}
TRIES=${TRIES:-6}
SRC=/root/out/fld_r11a/model.safetensors
I2="-i /root/.ssh/id_i2 -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=20"
H=root@175.155.64.145; P=19839
LOG=/root/ship_robust.log
say() { echo "[robust $(date -u +%H:%M:%S)] $*" >> "$LOG"; }

WANT=$(md5sum "$SRC" | awk '{print $1}')
say "source $(stat -c %s "$SRC") bytes, md5 $WANT, $N chunks, ${TMO}s per attempt"
rm -rf /root/ckr; mkdir -p /root/ckr
split -n "$N" -d -a 2 "$SRC" /root/ckr/c
ssh $I2 -p $P $H "rm -rf /root/ckr; mkdir -p /root/ckr"

send() {   # $1 = local chunk path
    local f="$1" b; b=$(basename "$f")
    local want; want=$(md5sum "$f" | awk '{print $1}')
    for try in $(seq 1 "$TRIES"); do
        if timeout "$TMO" scp $I2 -P $P -q "$f" "$H:/root/ckr/$b" 2>/dev/null; then
            local got; got=$(ssh $I2 -p $P $H "md5sum /root/ckr/$b 2>/dev/null | awk '{print \$1}'" 2>/dev/null)
            [ "$got" = "$want" ] && { say "$b ok (try $try)"; return 0; }
            say "$b arrived corrupt on try $try"
        else
            say "$b stalled past ${TMO}s on try $try -- reconnecting"
        fi
        ssh $I2 -p $P $H "rm -f /root/ckr/$b" 2>/dev/null
    done
    say "$b FAILED after $TRIES tries"
    return 1
}

T0=$(date +%s)
PIDS=""
for f in /root/ckr/c*; do send "$f" & PIDS="$PIDS $!"; done
RC=0
for p in $PIDS; do wait "$p" || RC=1; done
say "all chunk workers done in $(( $(date +%s) - T0 ))s (rc $RC)"
[ "$RC" = 0 ] || { say "SHIP_ROBUST_FAILED"; exit 1; }

ssh $I2 -p $P $H "cat /root/ckr/c* > /root/out/fld_r11a/model.safetensors && rm -rf /root/ckr"
GOT=$(ssh $I2 -p $P $H "md5sum /root/out/fld_r11a/model.safetensors | awk '{print \$1}'")
if [ "$GOT" = "$WANT" ]; then
    say "SHIP_ROBUST_OK in $(( $(date +%s) - T0 ))s"
    rm -rf /root/ckr /root/ckchunk /root/ckchunk2
else
    say "SHIP_ROBUST_MISMATCH want $WANT got $GOT"
fi
