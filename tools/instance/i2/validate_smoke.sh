#!/usr/bin/env bash
# Validate the smoke harness itself, once the GPU is free.  A preflight nobody has ever seen
# succeed is just another untested path -- and untested paths are what cost the last four nights.
set -u
until ! pgrep -f "[b]ash /root/night4b.sh" >/dev/null; do sleep 60; done
sleep 90                      # let the gate's last worker release the card
echo "[validate $(date -u +%m-%d_%H:%M:%S)] night4b finished; running the smoke harness"
SMOKE_ONLY=1 bash /root/night_run.sh /root/night4b.sh
echo "[validate $(date -u +%m-%d_%H:%M:%S)] SMOKE_VALIDATE_DONE rc=$?"
