#!/usr/bin/env bash
# after_pass.sh already did its swap (13:20), so writing DUSK_WRAP is no longer enough -- no
# consumer is left. Apply the wrapper straight to instance2's registry instead: every future
# collection and gate there builds the sparring dusknoir through "reg", so the wrap lands on
# the next process start with no further plumbing.
set -u
I2="-i /root/.ssh/id_i2 -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=20"
while [ ! -s /root/loop_dusk/mrl3/wrap.txt ]; do
    pgrep -f "[m]irror_chain3.sh" >/dev/null || exit 1
    sleep 60
done
W=$(head -1 /root/loop_dusk/mrl3/wrap.txt)
R5=lethal_now,spread_aim,clops_hold,energy_line,energy_focus
if [ "$W" = "def" ]; then
    ssh $I2 -p 19839 root@175.155.64.145 \
        "cd /root/ptcg/repo && PYTHONPATH=cg-lib python3 tools/adapters.py set dragapult_dusknoir --wrap planfilter:$R5 --note 'sparring mrl2_r5b + R5 deferral (v3 gate)' && echo planfilter:$R5 > /root/DUSK_WRAP"
    echo "[wrap_ship] def -> registry wrap applied on instance2"
else
    echo "[wrap_ship] bare -> nothing to ship"
fi
