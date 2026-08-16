#!/bin/bash
setsid --fork nohup bash /root/keepd.sh >/dev/null 2>&1 </dev/null
