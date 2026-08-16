#!/bin/bash
# Re-run the checkpointing A/B with nothing else on the box.
#
# The first attempt ran while several single-core Python jobs of mine were scanning the 2.5M-row
# dataset, and the same configuration came out 129s in one run and 166s in another -- a 28% swing
# that has nothing to do with the flag. Each setting is repeated so the spread is visible rather
# than assumed away.
set -u
cd /root/ptcg/repo
echo "load: $(uptime)"
run() {
  local tag="$1"; shift
  timeout 1800 python3 tools/instance/sft_teacher.py --domain-tokens \
    --action-vocab data/action_vocab_v39.json \
    --model unsloth/Qwen3-4B-Base --data data/sft/v39_dag005.jsonl.gz \
    --out /root/out/ck2 --limit 40000 --eval-n 0 --steps 30 --maxlen 896 \
    --save-steps 100000 --bsz 8 --accum 4 --group-by-length "$@" 2>&1 \
    | grep -E "^\[done\]|^\[peak\]" | sed "s/^/$tag /"
  rm -rf /root/out/ck2
}
for i in 1 2; do
  run "ON#$i"
  run "OFF#$i" --no-grad-ckpt
done
echo "load: $(uptime)"
echo "CKPT2 DONE"
