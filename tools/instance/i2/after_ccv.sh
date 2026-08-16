set -u
LOG=/root/after_ccv.log
say () { echo "[$(date -u +%H:%M:%S)] $*" >> $LOG; }
say "waiting for the causal-conv1d build"
while ! grep -qE "CCV_DONE|ERROR:" /root/fix_ccv.log 2>/dev/null; do sleep 10; done
if grep -q "ERROR:" /root/fix_ccv.log; then
  say "BUILD FAILED -- not starting the smoke test"
  grep -E "error|Error" /root/fix_ccv.log | tail -5 >> $LOG
  exit 1
fi
say "build finished; fast-path check:"
python3 -c "
from transformers.models.qwen3_5 import modeling_qwen3_5 as M
import torch
print(\"  torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), \"FAST PATH\", M.is_fast_path_available)
" >> $LOG 2>&1
say "smoke: 20 steps, bsz 8 x accum 1, index-target data"
cd /root && python3 sft_teacher.py --limit 4000 --steps 20 --bsz 8 --accum 1 \
    --out /root/out/smoke3 >> /root/smoke3.log 2>&1
say "smoke exit=$?"
grep -E "\[stack\]|\[load\]|\[peft\]|\[data\]|\[done\]|\[saved\]" /root/smoke3.log | tail -8 >> $LOG
grep -oE "20/20 \[[^]]*\]" /root/smoke3.log | tail -1 >> $LOG
say AFTER_CCV_DONE
