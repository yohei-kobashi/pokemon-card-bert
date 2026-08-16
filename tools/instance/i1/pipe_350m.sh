#!/bin/bash
# Full pipeline for LFM2.5-350M: download -> add tokens -> convert -> quantize -> tar -> measure
LOG=/root/pipe350.log
LIMIT=207257600   # 197.65625 MiB
echo "START $(date)" > $LOG
export CUDA_VISIBLE_DEVICES=

# 1) download base
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('LiquidAI/LFM2.5-350M', local_dir='/root/lfm2_350m', ignore_patterns=['*.gguf'])
print('DL_OK')
" >> $LOG 2>&1 || { echo DL_FAIL >> $LOG; exit 1; }

# 2) add domain tokens
python3 /root/add_tokens_arg.py /root/lfm2_350m /root/lfm2_350m_ext >> $LOG 2>&1 || { echo ADDTOK_FAIL >> $LOG; exit 1; }

# 3) convert to f16 gguf
cd /root/llama.cpp
python3 convert_hf_to_gguf.py /root/lfm2_350m_ext --outfile /root/lfm2_350m_ext_f16.gguf --outtype f16 >> $LOG 2>&1 || { echo CONVERT_FAIL >> $LOG; exit 1; }
echo "f16 size: $(stat -c%s /root/lfm2_350m_ext_f16.gguf) bytes" >> $LOG

# 4) quantize several levels
Q=/root/llama.cpp/build/bin/llama-quantize
for T in Q4_0 Q4_K_M Q3_K_M Q3_K_S; do
  $Q /root/lfm2_350m_ext_f16.gguf /root/lfm350ext.$T.gguf $T 8 >> $LOG 2>&1 \
    && echo "quant $T: $(stat -c%s /root/lfm350ext.$T.gguf) bytes" >> $LOG \
    || echo "quant $T FAILED" >> $LOG
done

# 5) build real tar.gz bundles (model + full llama_cpp + code) and measure vs limit
echo "=== BUNDLE tar.gz vs limit $LIMIT ===" >> $LOG
for T in Q3_K_S Q3_K_M Q4_0 Q4_K_M; do
  G=/root/lfm350ext.$T.gguf
  [ -f "$G" ] || continue
  D=/root/b350_$T; rm -rf $D; mkdir -p $D
  cp "$G" $D/model.gguf
  cp -r /root/subm_crustle/lm /root/subm_crustle/agents /root/subm_crustle/cg /root/subm_crustle/llama_cpp $D/ 2>/dev/null
  mkdir -p $D/decks; cp /root/subm_crustle/decks/crustle.csv $D/decks/ 2>/dev/null
  cp /root/subm_crustle/deck.csv /root/subm_crustle/main.py $D/ 2>/dev/null
  find $D -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
  tar czf /root/lfm350_$T.tar.gz -C $D .
  SZ=$(stat -c%s /root/lfm350_$T.tar.gz)
  echo "$T: tar.gz=$SZ bytes ($((SZ/1048576)) MiB) $([ $SZ -le $LIMIT ] && echo FITS || echo OVER_by_$(( (SZ-LIMIT)/1048576 ))MiB)" >> $LOG
  rm -rf $D
done
echo "PIPE350_DONE" >> $LOG
