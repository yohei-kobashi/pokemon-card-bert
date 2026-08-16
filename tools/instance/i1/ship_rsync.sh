#!/usr/bin/env bash
# Finish the last chunk with rsync, which resumes, instead of scp, which does not.
#
# c07 failed five scp attempts in a row: each stall was detected at 150 s and each retry started
# again from byte zero, so twelve minutes bought no net progress. That is the flaw in the chunked
# scp design -- the chunking existed only to bound what a restart throws away, and rsync bounds
# it to nothing by resuming from the partial file.
#
# --append-verify re-checksums the bytes already there before continuing, so a partial left by a
# killed transfer is trusted only after it is proven to match.
set -u
I2="-i /root/.ssh/id_i2 -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=20"
H=root@175.155.64.145; P=19839
LOG=/root/ship_robust.log
say() { echo "[rsync $(date -u +%H:%M:%S)] $*" >> "$LOG"; }

F=/root/ckr/c07
WANT=$(md5sum "$F" | awk '{print $1}')
say "c07 by rsync (resumes); want $WANT"

for try in $(seq 1 12); do
    timeout 120 rsync -a --partial --inplace --append-verify \
        -e "ssh $I2 -p $P" "$F" "$H:/root/ckr/c07" 2>/dev/null
    GOT=$(ssh $I2 -p $P $H "md5sum /root/ckr/c07 2>/dev/null | awk '{print \$1}'" 2>/dev/null)
    SZ=$(ssh $I2 -p $P $H "stat -c %s /root/ckr/c07 2>/dev/null || echo 0" 2>/dev/null)
    if [ "$GOT" = "$WANT" ]; then
        say "c07 complete after $try attempt(s)"
        break
    fi
    say "attempt $try: $SZ bytes of $(stat -c %s "$F") landed -- reconnecting (progress is kept)"
done

GOT=$(ssh $I2 -p $P $H "md5sum /root/ckr/c07 2>/dev/null | awk '{print \$1}'" 2>/dev/null)
[ "$GOT" = "$WANT" ] || { say "c07 STILL INCOMPLETE"; exit 1; }

ssh $I2 -p $P $H "cat /root/ckr/c* > /root/out/fld_r11a/model.safetensors && rm -rf /root/ckr"
SRC=/root/out/fld_r11a/model.safetensors
W=$(md5sum "$SRC" | awk '{print $1}')
G=$(ssh $I2 -p $P $H "md5sum $SRC | awk '{print \$1}'")
if [ "$W" = "$G" ]; then
    say "SHIP_ROBUST_OK (rsync finished the tail)"
    rm -rf /root/ckr /root/ckchunk /root/ckchunk2
else
    say "SHIP_ROBUST_MISMATCH want $W got $G"
fi
