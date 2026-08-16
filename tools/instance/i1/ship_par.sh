#!/usr/bin/env bash
# Move the 746 MB checkpoint in parallel chunks instead of one stream.
#
# MEASURED: three ssh handshakes take 8.7 s, so the round trip is 300-500 ms, and a single scp
# settles around 0.2-0.9 MB/s -- which is exactly what a 64 KB window over that RTT allows.
# The link itself is not the limit: while one scp was running, a separate 40 MB stream still got
# 0.9 MB/s of its own, so the two together beat either alone. That is the signature of a
# bandwidth-delay-product cap on ONE connection, and the fix is more connections.
#
# Chunks are verified individually and the reassembled file is checked against the source md5
# before anything is allowed to use it.
set -u
N=${N:-6}
SRC=/root/out/fld_r11a/model.safetensors
I2="-i /root/.ssh/id_i2 -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o BatchMode=yes"
H=root@175.155.64.145; P=19839
LOG=/root/ship_par.log
say() { echo "[shipp $(date -u +%H:%M:%S)] $*" >> "$LOG"; }

WANT=$(md5sum "$SRC" | awk '{print $1}')
SZ=$(stat -c %s "$SRC")
say "source $SZ bytes, md5 $WANT, $N parallel chunks"

rm -rf /root/ckchunk; mkdir -p /root/ckchunk
split -n "$N" -d "$SRC" /root/ckchunk/part_
ssh $I2 -p $P $H "rm -rf /root/ckchunk /root/out/fld_r11a/model.safetensors; mkdir -p /root/ckchunk /root/out/fld_r11a"

T0=$(date +%s)
for F in /root/ckchunk/part_*; do
    ( scp $I2 -P $P -q "$F" "$H:/root/ckchunk/$(basename $F)" && echo "ok $(basename $F)" >> "$LOG" \
      || echo "FAIL $(basename $F)" >> "$LOG" ) &
done
wait
say "all chunks sent in $(( $(date +%s) - T0 ))s"

ssh $I2 -p $P $H "cat /root/ckchunk/part_* > /root/out/fld_r11a/model.safetensors && rm -rf /root/ckchunk"
GOT=$(ssh $I2 -p $P $H "md5sum /root/out/fld_r11a/model.safetensors | awk '{print \$1}'")
say "remote md5 $GOT"
if [ "$GOT" = "$WANT" ]; then
    say "SHIP_PAR_OK in $(( $(date +%s) - T0 ))s total"
    rm -rf /root/ckchunk
else
    say "SHIP_PAR_MISMATCH -- leaving the chunks in place for a retry"
fi
# the small files are cheap; send them last so a rerun does not have to redo the big one
scp $I2 -P $P -q /root/out/fld_r11a/config.json /root/out/fld_r11a/tokenizer.json \
    /root/out/fld_r11a/tokenizer_config.json "$H:/root/out/fld_r11a/" && say "aux files sent"
