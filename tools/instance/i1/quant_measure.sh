#!/bin/bash
# wait for the rebuild, then quantize the token-extended LFM2.5-230M and report sizes
LOG=/root/quantmeasure.log
echo "START waiting for rebuild" > $LOG
for i in $(seq 1 90); do
  grep -q "REBUILD_EXIT=0" /root/rebuild.log 2>/dev/null && break
  grep -qE "REBUILD_EXIT=[1-9]" /root/rebuild.log 2>/dev/null && { echo "REBUILD FAILED" >> $LOG; tail -20 /root/rebuild.log >> $LOG; exit 1; }
  sleep 10
done
Q=/root/llama.cpp/build/bin/llama-quantize
echo "quantize binary: $(ls -la $Q 2>&1)" >> $LOG
F16=/root/lfm2_ext_f16.gguf
for T in Q4_K_M Q5_K_M Q6_K Q4_0; do
  OUT=/root/lfm2ext.$T.gguf
  echo "=== quantizing $T ===" >> $LOG
  $Q "$F16" "$OUT" "$T" 8 >> $LOG 2>&1 && echo "$T OK size=$(du -m "$OUT" | cut -f1)MB ($(stat -c%s "$OUT") bytes)" >> $LOG || echo "$T FAILED" >> $LOG
done
echo "=== FINAL SIZES ===" >> $LOG
ls -la /root/lfm2ext.*.gguf >> $LOG 2>&1
echo QUANTMEASURE_DONE >> $LOG
