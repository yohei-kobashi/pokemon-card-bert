#!/bin/bash
# Does length grouping actually save wall time? Measured over a FULL epoch of a small slice, so
# both orders process exactly the same samples and the only difference left is padding.
set -u
cp /tmp/sft_teacher.py /root/ptcg/repo/tools/instance/
cp /tmp/bench_sft.sh /root/ptcg/repo/tools/instance/; chmod +x /root/ptcg/repo/tools/instance/bench_sft.sh
cp /tmp/action_token.py /root/ptcg/repo/lm/
cd /root/ptcg/repo
grep -c "action-vocab" tools/instance/sft_teacher.py tools/instance/bench_sft.sh
run() {
  echo "=== $1 : $* ==="
  local n="$1"; shift
  timeout 3600 python3 tools/instance/sft_teacher.py --domain-tokens \
    --action-vocab data/action_vocab_v39.json \
    --model unsloth/Qwen3-4B-Base --data data/sft/v39_dag005.jsonl.gz \
    --out /root/out/pad_$n --limit 4000 --eval-n 0 --epochs 1 --maxlen 896 \
    --save-steps 100000 "$@" 2>&1 | grep -E "^\[done\]|^\[peak\]|^\[len\]|^\[action\]|^\[data\]|out of memory"
  rm -rf /root/out/pad_$n
}
run random --bsz 8 --accum 4
run sorted --bsz 8 --accum 4 --group-by-length
echo "PADTEST DONE"
