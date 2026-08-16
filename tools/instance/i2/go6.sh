#!/bin/bash
cd /root && setsid --fork nohup bash /root/night_run.sh /root/night6.sh >/root/night6_run.log 2>&1 </dev/null
