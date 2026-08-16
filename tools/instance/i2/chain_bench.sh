#!/bin/bash
# Wait for the model download and the data mix, then benchmark. Both prerequisites are checked
# for their SUCCESS marker, not merely for the process being gone -- a download that died leaves
# a partial cache that only fails later, inside the first bench config.
set -u
say(){ echo "[chain $(date -u +%H:%M:%S)] $*"; }
say "waiting for download + mix"
for i in $(seq 1 240); do
  grep -q "DL DONE" /root/dl4b.log 2>/dev/null && grep -q "MIXDONE" /root/mk_mix.log 2>/dev/null && break
  sleep 30
done
grep -q "DL DONE" /root/dl4b.log || { say "STOP: model download never finished"; exit 1; }
grep -q "MIXDONE" /root/mk_mix.log || { say "STOP: mix never finished"; exit 1; }
ls -la /root/ptcg/repo/data/sft/v39_dag005.jsonl.gz
grep "dagger share\|-> /root" /root/mk_mix.log | tail -2
say "prerequisites OK -> bench"
/root/ptcg/repo/tools/instance/bench_sft.sh
say "CHAIN DONE"
