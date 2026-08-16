#!/bin/bash
setsid --fork nohup bash /root/ship_rsync.sh >/dev/null 2>&1 </dev/null
