#!/usr/bin/env bash
# Recover fld_r41b (the marnie/alakazam-strong old champion) from instance2 -> /root/out.
# rsync with retries -- this link stalls at 40-90 kB/s sometimes (ckptd.sh header).
set -u
I2="-i /root/.ssh/id_i2 -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=20"
H=root@175.155.64.145; P=19839
mkdir -p /root/out/fld_r41b
for try in $(seq 1 30); do
    timeout 300 rsync -a --partial --inplace --append-verify -e "ssh $I2 -p $P" \
        "$H:/root/out/fld_r41b/" /root/out/fld_r41b/ && break
    echo "[pull41b] attempt $try stalled, redialling" >> /root/pull_r41b.log
    sleep 5
done
ls -la /root/out/fld_r41b/ >> /root/pull_r41b.log
echo "PULL41B_DONE" >> /root/pull_r41b.log
