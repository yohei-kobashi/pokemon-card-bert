#!/bin/bash
setsid --fork nohup bash /root/ckptd.sh >/dev/null 2>&1 </dev/null
