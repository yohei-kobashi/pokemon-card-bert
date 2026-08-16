#!/bin/bash
setsid --fork nohup bash /root/restart_at_boundary.sh >/dev/null 2>&1 </dev/null
