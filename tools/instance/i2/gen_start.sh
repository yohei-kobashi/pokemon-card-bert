#!/usr/bin/env bash
# Wait for the DeBERTa champion to finish copying, verify it, then start generating.
#
# The gate is the checksum instance1 computed on the source, not the file's existence: a partial
# safetensors passes `[ -s ]` and then fails at load time, which on this project has repeatedly
# meant discovering the problem hours later instead of at the start.
set -u
WANT=2baf33fb0bc845a77af5191089dba3d4
F=/root/out/fld_r11a/model.safetensors
LOG=/root/gend.log
say() { echo "[start $(date -u +%m-%d_%H:%M:%S)] $*" >> "$LOG"; }
say "waiting for the champion to finish copying (want $WANT)"
for i in $(seq 1 360); do
    if [ -s "$F" ]; then
        GOT=$(md5sum "$F" | awk '{print $1}')
        if [ "$GOT" = "$WANT" ]; then
            say "checkpoint verified after $((i * 20))s -- starting the generator"
            exec setsid --fork nohup bash /root/gend2.sh >/dev/null 2>&1 </dev/null
        fi
    fi
    sleep 20
done
say "STOP: the checkpoint never completed"
