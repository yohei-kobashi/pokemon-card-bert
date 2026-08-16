#!/bin/bash
setsid --fork nohup bash /root/accel_gate.sh >/dev/null 2>&1 </dev/null
