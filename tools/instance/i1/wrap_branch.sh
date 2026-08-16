#!/usr/bin/env bash
# The running chain predates the bare-mode reward fix. If step 0 chooses "def" the two scripts
# are byte-identical in behaviour and the running one may continue; if it chooses "bare" the
# running one would wrongly keep the five rules out of the reward, so it is restarted on the
# fixed script -- wrap.txt is cached, so the restart skips straight past step 0.
set -u
while [ ! -s /root/loop_dusk/mrl3/wrap.txt ]; do
    pgrep -f "[m]irror_chain3.sh" >/dev/null || exit 1
    sleep 60
done
W=$(head -1 /root/loop_dusk/mrl3/wrap.txt)
mv /root/mirror_chain3.sh.new /root/mirror_chain3.sh
if [ "$W" = "bare" ]; then
    echo "[wrap_branch] bare -> restarting the chain on the fixed script"
    pkill -f "[m]irror_chain3.sh" || true
    sleep 3
    pkill -f "lm_mirror_log.py --model hf:" || true
    sleep 5
    cd /root
    setsid nohup bash /root/mirror_chain3.sh >> /root/mirror_chain3.log 2>&1 < /dev/null &
else
    echo "[wrap_branch] def -> running chain already behaves identically; script swapped for the record"
fi
