#!/bin/bash
setsid --fork nohup bash /root/statusd.sh >/dev/null 2>&1 </dev/null
