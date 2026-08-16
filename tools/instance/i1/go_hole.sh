#!/bin/bash
setsid --fork nohup bash /root/hole_launch.sh >/root/hole_launch.log 2>&1 </dev/null
