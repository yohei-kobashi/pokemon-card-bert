#!/bin/bash
# Fires when the SFT process exits, so the GPU is never shared with training.
# PID is pinned (not pgrep) because `pgrep -f sft_teacher` also matches the shell running THIS
# script -- that pattern already killed its own ssh command once this session.
PID=15761
cd /root
while kill -0 $PID 2>/dev/null; do sleep 60; done
sleep 45
ADP=/root/out/teacher9b
if [ ! -f "$ADP/adapter_model.safetensors" ]; then
  ADP=$(ls -d /root/out/teacher9b/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
  echo "final adapter missing; falling back to $ADP" > /root/eval_fallback.txt
fi
echo "adapter=$ADP" > /root/eval_stage.txt
python3 eval_teacher.py --adapter "$ADP" --limit 60 > /root/eval_smoke.log 2>&1
if ! grep -q "FULL COVERAGE" /root/eval_smoke.log; then
  echo "SMOKE FAILED - stopping" >> /root/eval_stage.txt; exit 1
fi
echo "smoke ok, running AFTER" >> /root/eval_stage.txt
python3 eval_teacher.py --adapter "$ADP" --dump /root/teacher_dist.jsonl.gz \
        --pad-check 64 > /root/eval_after.log 2>&1
echo "AFTER done, running BEFORE (base)" >> /root/eval_stage.txt
python3 eval_teacher.py > /root/eval_before.log 2>&1
echo "ALL DONE" >> /root/eval_stage.txt
