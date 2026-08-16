#!/usr/bin/env bash
# Delete dead artifacts on the vast box. Code is never touched: everything here is a model
# checkpoint, a quantised export, generated training data, or a package cache.
#
# KEEP (do not add to this list):
#   /root/ptcg/repo/{tools,lm,agents,cg-lib,decks,*.py}   the code
#   /root/*.py /root/*.sh /root/repo_sync2.tgz            scripts + a repo archive
#   /root/out/rerank_gte_v35 /root/out/rerank_gte_v36     the two live models
#   /root/onnx_v35 /root/onnx_v36 + submissions/*.tar.gz  the deploy artifacts
#   /root/data/rerank/*_v36*  /root/ptcg/repo/data/rerank/*_v36w*   the ablation arms
#   /root/data/rerank/*_v2*                               v35's training data (bundle rebuilds)
#   /root/ptcg/repo/data/selfplay/{curengine_0724,v34_full,meta_topup_0723,mega_starmie_v2}
#                                                         the SOURCE logs -- data is rebuildable
#                                                         from these, so they are the backup
#   /root/.cache/huggingface                              holds trust_remote_code modeling code
#                                                         for gte-reranker; deleting it mid-run
#                                                         could break training
set -u
BEFORE=$(df --output=avail -BM / | tail -1 | tr -d ' M')

DEAD=(
  # superseded ONNX experiments -- the shipped exports live in onnx_v35 / onnx_v36
  /root/onnx
  # Qwen3.5 GGUFs: the whole decoder deploy path died on the 197.66 MiB tarball cap
  /root/sft.f16.gguf /root/sft2.f16.gguf /root/sftv2.f16.gguf /root/sft.Q8_0.gguf
  # Qwen LoRA adapters + merged policies for the OLD-format RL run, discarded on the reformat
  /root/ptcg/repo/out/lora_v34 /root/ptcg/repo/out/lora_v34_sftbase /root/ptcg/repo/out/lora_v35
  /root/ptcg/repo/out/rl/sft_merged /root/ptcg/repo/out/rl/sft_v2_merged
  # LFM2-350M: measured 31pt under the engine baseline, abandoned
  /root/lfm2_350m /root/lfm2_350m_ext /root/lfm2_350m_ext_f16.gguf /root/lfm2_ext_f16.gguf
  /root/out/lfm2_350m_sft /root/data/sft_lfm2
  # decoder SFT corpora (Qwen/LFM2 prompt format) -- no decoder is being trained
  /root/ptcg/repo/data/sft /root/ptcg/repo/data/sft_v2
  # deck perturbation: refuted by measurement (0.0% label change), must never be trained on
  /root/data/rerank/perturb_0727_v36.rerank.jsonl.gz
  /root/ptcg/repo/data/selfplay/perturb_0727
  # rerank builds in prompt formats no live model uses (full/structured glossary, sorted DECK)
  /root/data/rerank/curengine_0724.rerank.jsonl.gz
  /root/data/rerank/curengine_0724_mp.rerank.jsonl.gz
  /root/data/rerank/curengine_0724_none.rerank.jsonl.gz
  /root/data/rerank/curengine_0724_rem.rerank.jsonl.gz
  # package cache, rebuilt on demand
  /root/.cache/pip
)

for p in "${DEAD[@]}"; do
  if [ -e "$p" ]; then
    sz=$(du -shx "$p" 2>/dev/null | cut -f1)
    rm -rf "$p" && printf "  deleted %8s  %s\n" "$sz" "$p"
  else
    printf "  absent           %s\n" "$p"
  fi
done

AFTER=$(df --output=avail -BM / | tail -1 | tr -d ' M')
echo
echo "free ${BEFORE}M -> ${AFTER}M  (+$((AFTER - BEFORE)) MB)"
df -h / | tail -1
echo
echo "=== survivors that matter ==="
du -shx /root/out/rerank_gte_v35 /root/out/rerank_gte_v36 /root/onnx_v35 /root/onnx_v36 \
        /root/data/rerank /root/ptcg/repo/data/rerank /root/ptcg/repo/data/selfplay \
        /root/ptcg/repo/submissions /root/repo_sync2.tgz 2>/dev/null
echo
echo "=== code intact? ==="
ls /root/ptcg/repo/tools/*.py 2>/dev/null | wc -l | xargs echo "  tools/*.py:"
ls /root/ptcg/repo/lm/*.py 2>/dev/null | wc -l | xargs echo "  lm/*.py:"
ls /root/ptcg/repo/agents/*.py 2>/dev/null | wc -l | xargs echo "  agents/*.py:"
ls /root/ptcg/repo/decks/*.csv 2>/dev/null | wc -l | xargs echo "  decks/*.csv:"
