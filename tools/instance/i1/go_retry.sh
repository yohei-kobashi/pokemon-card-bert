#!/bin/bash
setsid --fork nohup bash /root/ship_retry.sh >/dev/null 2>&1 </dev/null
