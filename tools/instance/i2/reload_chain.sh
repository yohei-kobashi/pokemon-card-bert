#!/bin/bash
set -u
for p in $(ps -eo pid,args | grep "[c]hain_cf.sh" | awk '{print $1}'); do kill $p 2>/dev/null; done
sleep 3
cp /tmp/chain_cf.sh /root/ptcg/repo/tools/instance/chain_cf.sh
chmod +x /root/ptcg/repo/tools/instance/chain_cf.sh
cd /root/ptcg/repo
setsid nohup tools/instance/chain_cf.sh > /dev/null 2>&1 < /dev/null &
sleep 15
echo "chain procs: $(ps -eo args | grep -c '[c]hain_cf.sh')"
tail -2 /root/chain_cf.log
