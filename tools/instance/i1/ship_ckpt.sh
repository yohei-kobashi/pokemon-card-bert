#!/usr/bin/env bash
set -u
I2="-i /root/.ssh/id_i2 -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o BatchMode=yes"
H=root@175.155.64.145; P=19839
ssh $I2 -p $P $H "mkdir -p /root/out/fld_r11a"
rsync -a --info=progress2 -e "ssh $I2 -p $P" /root/out/fld_r11a/ $H:/root/out/fld_r11a/ \
  2>/dev/null || scp $I2 -P $P -r /root/out/fld_r11a/* $H:/root/out/fld_r11a/
md5sum /root/out/fld_r11a/* | sort > /root/ckpt_local.md5
ssh $I2 -p $P $H "md5sum /root/out/fld_r11a/* | sort" > /root/ckpt_remote.md5
diff -q /root/ckpt_local.md5 /root/ckpt_remote.md5 && echo SHIP_OK || echo SHIP_MISMATCH
