#!/usr/bin/env bash
# Keep instance2's copy of the dusknoir champion current, because the champion now moves.
#
# MIN_GAIN=0.0 means a round adopts on any positive delta, and round 23 already moved the
# champion from fld_r11a to fld_r23b (+2.67 +- 1.46). Generating traces from a stale checkpoint
# would make them off-policy for the very model they are meant to train, which is the reason the
# dusknoir side is the DeBERTa at all.
#
# TRANSFER METHOD, learned the hard way today: rsync with a stall timeout, not scp.
#   scp does not resume, so killing a stalled stream throws away everything it had sent -- chunk
#   c07 failed five 150 s attempts in a row and made no net progress in twelve minutes. The same
#   chunk finished on the FIRST rsync attempt in 31 s. Some connections on this path settle at
#   40-90 kB/s and never recover; the remedy is to hang up and redial, and rsync makes redialling
#   free because the partial file is kept and re-verified.
set -u
LOG=/root/ckptd.log
I2="-i /root/.ssh/id_i2 -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=20"
H=root@175.155.64.145; P=19839
STOP=$(date -u -d "${STOP_AFTER:-2026-08-16T23:00:00Z}" +%s)
say() { echo "[ckptd $(date -u +%m-%d_%H:%M:%S)] $*" >> "$LOG"; }
say "watching the registry; will keep instance2's champion in step"

ship_one() {   # $1 = local dir, $2 = basename, $3 = file
    local d="$1" n="$2" f="$3" want got sz
    want=$(md5sum "$d/$f" | awk '{print $1}')
    for try in $(seq 1 15); do
        timeout 180 rsync -a --partial --inplace --append-verify \
            -e "ssh $I2 -p $P" "$d/$f" "$H:/root/out/$n/$f" 2>/dev/null
        got=$(ssh $I2 -p $P $H "md5sum /root/out/$n/$f 2>/dev/null | awk '{print \$1}'" 2>/dev/null)
        [ "$got" = "$want" ] && { [ "$try" -gt 1 ] && say "  $f ok after $try attempts" ; return 0; }
        sz=$(ssh $I2 -p $P $H "stat -c %s /root/out/$n/$f 2>/dev/null || echo 0" 2>/dev/null)
        say "  $f: $sz/$(stat -c %s "$d/$f") after attempt $try -- redialling (progress kept)"
    done
    return 1
}

LAST=""
while [ "$(date -u +%s)" -lt "$STOP" ]; do
    CUR=$(python3 - <<'PY' 2>/dev/null
import json, os
r = json.load(open("/root/ptcg/repo/models/adapters.json"))
t = (r["decks"]["dragapult_dusknoir"]["target"] or "").partition(":")[2]
print(t if t.startswith("/") else os.path.join("/root/out", t))
PY
)
    N=$(basename "${CUR:-}")
    if [ -z "$CUR" ] || [ ! -s "$CUR/model.safetensors" ]; then
        say "registry points at ${CUR:-nothing} which has no weights -- waiting"; sleep 300; continue
    fi
    if [ "$N" = "$LAST" ]; then sleep 180; continue; fi

    say "champion is now $N -- shipping"
    ssh $I2 -p $P $H "mkdir -p /root/out/$N" 2>/dev/null
    OK=1
    for F in model.safetensors config.json tokenizer.json tokenizer_config.json; do
        [ -f "$CUR/$F" ] || continue
        ship_one "$CUR" "$N" "$F" || { say "FAILED to ship $F"; OK=0; break; }
    done
    if [ "$OK" = 1 ]; then
        # the pointer is written LAST, so the generator never sees a name whose weights are
        # still arriving
        ssh $I2 -p $P $H "echo /root/out/$N > /root/out/champion.txt" 2>/dev/null \
            && { say "$N delivered and pointed to"; LAST="$N"; } \
            || say "shipped $N but could not write the pointer"
    fi
    sleep 60
done
say "CKPTD_DONE (past stop time)"
