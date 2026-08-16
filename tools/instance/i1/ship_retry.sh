#!/usr/bin/env bash
# The three slow streams did not recover when the fast ones finished, so re-open them: fresh
# connections, and each remaining chunk split again so there are more of them.
set -u
I2="-i /root/.ssh/id_i2 -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o BatchMode=yes"
H=root@175.155.64.145; P=19839
LOG=/root/ship_par.log
say() { echo "[retry $(date -u +%H:%M:%S)] $*" >> "$LOG"; }
for X in $(ps -eo pid,args | awk "/[s]cp .*part_0[024]/{print \$1}"); do kill -9 $X 2>/dev/null; done
sleep 2
mkdir -p /root/ckchunk2
rm -f /root/ckchunk2/*
for C in 00 02 04; do split -n 2 -d "/root/ckchunk/part_$C" "/root/ckchunk2/p${C}_"; done
ssh $I2 -p $P $H "rm -f /root/ckchunk/part_00 /root/ckchunk/part_02 /root/ckchunk/part_04; mkdir -p /root/ckchunk2; rm -f /root/ckchunk2/*"
say "retrying 00/02/04 as $(ls -1 /root/ckchunk2 | wc -l) streams"
T0=$(date +%s)
for F in /root/ckchunk2/*; do
    ( scp $I2 -P $P -q "$F" "$H:/root/ckchunk2/$(basename $F)" && say "ok $(basename $F)" || say "FAIL $(basename $F)" ) &
done
wait
say "retry streams done in $(( $(date +%s) - T0 ))s"
ssh $I2 -p $P $H "cd /root/ckchunk2 && cat p00_00 p00_01 > /root/ckchunk/part_00 && cat p02_00 p02_01 > /root/ckchunk/part_02 && cat p04_00 p04_01 > /root/ckchunk/part_04 && rm -rf /root/ckchunk2"
ssh $I2 -p $P $H "cat /root/ckchunk/part_* > /root/out/fld_r11a/model.safetensors"
WANT=$(md5sum /root/out/fld_r11a/model.safetensors | awk "{print \$1}")
GOT=$(ssh $I2 -p $P $H "md5sum /root/out/fld_r11a/model.safetensors | awk \"{print \\\$1}\"")
say "want $WANT got $GOT"
[ "$WANT" = "$GOT" ] && { say SHIP_PAR_OK; ssh $I2 -p $P $H "rm -rf /root/ckchunk"; rm -rf /root/ckchunk /root/ckchunk2; } || say SHIP_PAR_MISMATCH
