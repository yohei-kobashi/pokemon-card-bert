#!/usr/bin/env bash
set -u
I2="-i /root/.ssh/id_i2 -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o BatchMode=yes"
H=root@175.155.64.145; P=19839
ssh $I2 -p $P $H "rm -rf /root/out/fld_r11a; mkdir -p /root/out/fld_r11a"
T0=$(date +%s)
scp $I2 -C -P $P /root/out/fld_r11a/* $H:/root/out/fld_r11a/
echo "transfer took $(( $(date +%s) - T0 ))s"
L=$(md5sum /root/out/fld_r11a/model.safetensors | awk "{print \$1}")
R=$(ssh $I2 -p $P $H "md5sum /root/out/fld_r11a/model.safetensors | awk \"{print \\\$1}\"")
echo "local  $L"; echo "remote $R"
[ "$L" = "$R" ] && echo SHIP_OK || echo SHIP_MISMATCH
