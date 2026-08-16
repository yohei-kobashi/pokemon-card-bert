#!/bin/bash
set -u
for p in $(ps -eo pid,args | grep "[s]ft_teacher.py" | awk '{print $1}'); do kill $p 2>/dev/null; done
for p in $(ps -eo pid,args | grep "[b]ench_sft.sh" | awk '{print $1}'); do kill $p 2>/dev/null; done
sleep 8
mv -f /root/bench_sft.log /root/bench_sft.old2.log 2>/dev/null
setsid nohup /root/ptcg/repo/tools/instance/bench_sft.sh > /dev/null 2>&1 < /dev/null &
sleep 25
echo "bench procs: $(ps -eo args | grep -c '[b]ench_sft.sh')"
echo "--- new log ---"
cat /root/bench_sft.log
