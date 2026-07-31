# Plan B — a large teacher, RL'd, then distilled into the shippable cross-encoder

**Goal.** Train a big decoder (SFT → Stage-B RL) that plays well, then distil its per-decision
candidate distribution into the 149M `gte-reranker-modernbert-base` cross-encoder that actually
ships. The teacher never ships, so its size and architecture are free choices.

**Runs on a SECOND instance, in parallel with the reranker work on instance 1.**

---

## Two premise corrections, both measured

### 1. Do NOT use 4-bit QLoRA — Unsloth advises against it for Qwen3.5

From Unsloth's own Qwen3.5 fine-tuning guide:

> "It is not recommended to do QLoRA (4-bit) training on the Qwen3.5 models" (quantization
> differences), and "MoE QLoRA 4-bit is not recommended due to BitsandBytes limitations".

And it is not needed, because bf16 LoRA already fits (Unsloth's own table):

    0.8B 3 GB | 2B 5 GB | 4B 10 GB | 9B 22 GB | 27B 56 GB

So the answer to the memory worry is **bf16 LoRA, not 4-bit**:

| card | VRAM | teacher that fits with headroom |
|---|---|---|
| 4090 | 24 GB | **Qwen3.5-4B** (10 GB) — lots of room for rollout batching |
| 5090 | 32 GB | **Qwen3.5-9B** (22 GB) — workable |
| 24 GB + 9B | 22 of 24 GB | too tight to also run rollouts; avoid |

Since the teacher's quality is the whole point and the instance is being rented anyway,
**5090 (32 GB) + Qwen3.5-9B** is the recommendation; fall back to 4B on a 24 GB card.

For context on why the vocab still matters: Qwen3.5's vocab is **248,320** with tied embeddings,
so a naive `batch × seq × 248320` logits tensor is 1.27 GB at batch 8 × 320 tokens and the
gradient doubles it. Unsloth's fused cross-entropy never materialises it — that is a real reason
to use Unsloth here, independent of quantization.

What IS binding: **rollout throughput for RL, and CPU cores** (the game engine is CPU-bound, and
branch playouts are pure CPU). Spec the instance for cores + inference speed, not VRAM.

### 2. Qwen3.5 IS supported — earlier caution retracted, with three real caveats

Unsloth documents Qwen3.5 SFT **and** GRPO, for 0.8B / 2B / 4B / 9B / 27B / 35B-A3B / 122B-A10B.
My earlier "architecture support unverified" objection was wrong and is withdrawn: the
gated-DeltaNet hybrid has custom Mamba Triton kernels in Unsloth. Also, the prior Qwen3.5 failure
in this project was **GGUF→HF conversion**, which was a DEPLOY-path problem — the teacher never
ships, so it does not apply here at all.

Three caveats that do matter and belong in Phase 0:

1. **`transformers v5` is required.** Instance 1 already has 5.14.1, so the constraint is only
   about not pinning something older on the new box.
2. **GRPO must run with vLLM fast inference DISABLED** (use Unsloth inference), and vLLM ≥ 0.17.0
   if used at all — 0.16.0 does not support Qwen3.5. This directly sets the RL cost model, so
   measure decisions/s in that exact configuration, not with vLLM.
3. **Chat-template / EOS mismatch degrades inference.** Our prompt is a raw completion format
   (`[ACT]\nDECK[...] ... || SEL ... :: 0=… 1=…`), not a chat turn. Decide explicitly whether to
   wrap it in the chat template or train raw, and keep training and inference identical.

Qwen3.5 is a unified VLM, but Unsloth exposes selective layer targeting (vision / language /
attention / MLP), so train language layers only and leave the vision tower alone.

---

## The dependency that decides whether Plan B can pay

Distillation cannot give the student capacity it does not have. The student's size is **fixed by
the Kaggle 197.66 MiB submission cap** (149M at INT8 + pruned vocab = the current 128 MiB bundle;
a 395M student cannot fit). So:

* If `tools/rank_probe.py` shows the cross-encoder CAN be pushed above its current
  +0.0469 by supervised fitting → the student has headroom, and a better teacher fills it.
  **Plan B pays directly.**
* If it cannot → the student is at its representation limit, and a 4B teacher distils into the
  same ceiling. **Plan B is then diagnostic only** — still valuable (see below), but not a path to
  a stronger submission without changing the prompt or the student.

Interim signal (2026-07-30): rank_probe epoch 0 moved +0.0469 → +0.0442, i.e. nothing. If epoch 1
confirms, assume the representation-limit branch and re-scope.

**Plan B is worth running even on the pessimistic branch**, for one reason nothing else gives us:
it measures the TASK's ceiling. If a 4B model with RL beats engine_v2 decisively, the task is
winnable and the bottleneck is the student. If the 4B model ALSO plateaus at parity, the
bottleneck is the data/prompt/objective and no student change helps. That answer is worth the
GPU-days by itself.

---

## Phases

### Phase 0 — day-0 smoke tests (2–4 h, no training)

Every item is a gate; stop and re-decide if one fails.

1. `unsloth.FastLanguageModel.from_pretrained(...)` in **bf16 LoRA** (not 4-bit) loads and a
   20-step fit runs; language layers only, vision tower untouched. Time the first step separately
   — Unsloth warns the custom Mamba Triton kernels can be slow to compile, so a slow step 1 is
   compilation, not throughput.
2. **Tokenise a real prompt** with the teacher's tokenizer and record the length. Our prompt is
   built for a domain vocabulary; without it `c1152`/`d_crustle` split into several BPE pieces.
   This number sets `max_seq_length` and therefore throughput. Measure, do not guess.
3. **Do NOT add domain tokens.** Measured on the student: 3,087 added tokens collapsed to
   near-identical vectors (cos +0.998), and re-initialising them reset accuracy 65.7% → 24.6%.
   Adding them here would mean training a 248k × hidden tied embedding. Accept the longer prompt.
4. **Fix the prompt/EOS contract.** Raw completion vs chat template — pick one, and make the
   inference path byte-identical to the training path. Unsloth lists template/EOS mismatch as a
   known cause of degraded inference, and this project has already shipped a prompt-format drift
   bug once (`bundle-drops-id-segment`).
5. Save + reload a merged adapter and run inference from the reload. Prove the round trip on
   day 0, not at the end.
6. Measure decisions/s **with vLLM disabled / Unsloth inference**, at the real prompt length →
   the RL cost model. Measuring with vLLM would flatter a configuration GRPO cannot use.

### Phase 1 — SFT the teacher

* Data: re-render existing engine_v2 games in decoder format (prompt + the chosen candidate
  index as a ~1-token target). Reuse the current `PROMPT_FMT` so the student and teacher see the
  same state text — that is what makes distillation a clean listwise transfer later.
* Config: unsloth 4-bit NF4 (double quant), LoRA r=32 on q/k/v/o/gate/up/down, bf16 compute,
  gradient checkpointing, paged 8-bit AdamW, fused CE, `max_seq_length` from Phase 0.
* **Gate:** held-out top-1 on engine_v2's actions must clearly beat the reranker's **69.7%**
  (v37, 2,000 rows). If it does not, this teacher is not a better teacher and the rest is moot.

### Phase 2 — does the teacher actually PLAY? (before any RL)

* Play the live-weighted protocol vs engine_v2. The bar is the reranker's current gate
  (−5.65pt at r8) and engine_v2 parity (0).
* Re-measure decisions/s in the real loop.
* **Gate:** teacher ≥ reranker in play, not just in top-1. Top-1 on engine_v2 actions rewards
  imitating engine_v2, which is not the objective.

### Phase 3 — Stage-B RL on the teacher

* Stage B is the right stage by design: opponents reweighted to the live meta *before* any
  specialisation, because self-play is 48pt wrong about alakazam.
* Reuse `tools/rl_*.py`. NOTE: those were ported from decoder+LoRA to the cross-encoder and the
  decoder versions were **quarantined, not deleted** — resurrect from quarantine rather than
  rewrite, and re-check that the prompt renderer is the current one (a previous port shipped the
  wrong prompt).
* Keep the value-free design: GRPO + RAE matchup baseline. No value net in the reward.
* Budget from Phase 0's decisions/s. Expect a round to cost more than the 52 min the 149M model
  takes; scale games/round down rather than skipping the gate.

### Phase 4 — distil teacher → student

The mechanics are unusually clean, because a decoder that answers with a candidate INDEX gives
the whole distribution in one forward:

```
one teacher forward per decision
  -> logits, restricted to the candidate-index tokens
  -> softmax = soft target over the SAME candidate set the cross-encoder scores
  -> student loss = listwise cross-entropy  (exactly train_rerank.py's parameterisation)
```

* No new machinery on the student side.
* Cheap: at ~10 decisions/s, 200–300k labelled decisions is ~8 h of one GPU.
* Mix in the 8-playout Q targets we already have (`/root/out/branch8.jsonl.gz`, 99,139 points,
  59,974 usable) as a second target head or as extra rows; the two signals are complementary —
  teacher = "what a strong player prefers", Q = "what actually wins from here".
* **Gate:** student's held-out `E[Q(top) − mean Q(others)]` must rise above +0.0469, and the
  live-weighted play gate must beat −5.65pt.

### Phase 5 — deploy

Student size is unchanged, so the existing path applies: weight-only INT8 (blk128) + vocab
pruning → ~128 MiB bundle under the 197.66 MiB cap, ONNX runtime, time bank with engine_v2
fallback.

---

## Instance spec

Driven by throughput, not VRAM:

* **5090 (32 GB) + Qwen3.5-9B bf16 LoRA (22 GB)** is the pick. A 24 GB 4090 forces either 4B
  (10 GB, comfortable) or a 9B run with ~2 GB spare, which cannot also host rollouts. Do not pay
  for an A100 80 GB — the memory is unneeded and it is slower for bf16 dense compute than a 4090.
* **Many CPU cores matter more than extra VRAM.** The engine and the branch playouts are CPU-only;
  instance 1 sits at 165 of 256 cores during a fleet sweep. Aim for ≥64 cores.
* ≥100 GB disk (teacher checkpoints + rollouts).

## Kill criteria, stated up front

| stop if | because |
|---|---|
| Phase 0 item 1 or 5 fails | switch to a dense Qwen3-8B; do not debug kernels for a throwaway teacher |
| Phase 1 top-1 ≤ 69.7% | the teacher is not better than what we already have |
| Phase 2 play ≤ reranker | top-1 was imitation, not strength |
| Phase 4 student does not exceed +0.0469 | representation limit confirmed; the teacher's value is diagnostic only |

## Cross-references (memory)

`qwen35-2b-vast-training`, `lfm2-350m-sft-run`, `domain-token-embedding-degeneracy`,
`submission-size-limit-lfm2-pivot`, `rl-design-value-free` (Stage B, value-free),
`rl-stack-cross-encoder` (quarantined decoder tools), `live-alakazam-beats-us`,
`live-weighted-eval-protocol`, `shaping-potential-refuted` (the +0.0469 / +0.1638 numbers),
`rerank-deploy-quantization-and-speed`.
