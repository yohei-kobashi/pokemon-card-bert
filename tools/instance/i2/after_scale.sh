#!/bin/bash
while ! grep -q "SCALETEST DONE" /root/scaletest.log 2>/dev/null; do sleep 20; done
/root/ckpttest.sh > /root/ckpttest.log 2>&1
