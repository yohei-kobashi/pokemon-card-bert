#!/bin/bash
setsid --fork nohup bash /root/go_second_gate.sh >/dev/null 2>&1 </dev/null
