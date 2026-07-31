# Quarantined: the abandoned decoder deploy path

These files implemented the Qwen3.5-0.8B / LFM2 decoder agent and its llama.cpp GGUF
deployment. That path is dead on two independent counts:

* **Size** — the Qwen bundle was 523 MB against a 197.65625 MiB tarball cap
  (`submission-size-limit-lfm2-pivot`).
* **Strength** — every decoder measured below the engine_v2 baseline; LFM2-350M by 31pt
  (`lm-below-engine-baseline`).

The shipped agent is a `gte-reranker-modernbert-base` cross-encoder scored through
`lm/rerank_scorer.py` (ONNX, INT8, vocab-pruned) and trained by `tools/train_rerank.py`.

They are MOVED, not deleted: this repository is not under version control, so a delete is
unrecoverable. Being outside the import path is enough — nothing can `import sft_train_eval`
by accident, which is the actual failure mode being prevented. Take one back only after
checking it against the current prompt format and model class; every one of them assumes
`AutoModelForCausalLM` + LoRA adapters and the pre-2026-07-27 `glossary="full"` prompt.

| file | what it was |
|---|---|
| `sft_train_eval.py` | decoder SFT trainer + `ScoringModel` (mean-token-logprob scoring) |
| `build_sft.py` | decoder SFT data builder (superseded by `build_rerank.py`) |
| `calibrate_value_margin.py` | value-scorer margin calibration for `build_sft` |
| `init_adapter.py` | fresh LoRA adapter for the RL policy |
| `merge_adapter.py` | merge a LoRA back into the base |
| `rl_infer.py` | batched rollout server with per-adapter `set_adapter` grouping |
| `rl_temp_smoke.py` | sampling-temperature smoke test for the decoder policy |
| `rl_export_gguf.sh` | merge -> GGUF -> imatrix -> Q6_K/Q8_0 for llama.cpp |
| `kaggle_speed_notebook.py` | llama.cpp CPU speed notebook |
| `lm_scorer.py` | was `lm/scorer.py`: the llama.cpp `LlamaScorer` |
