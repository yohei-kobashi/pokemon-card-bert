#!/usr/bin/env python3
"""How much of the SFT compute goes into logits nobody looks at.

The answer is one or two tokens; the model computes a distribution over the whole vocabulary at
every one of ~800 positions and the loss then masks all but the answer. This prices that.
"""
from transformers import AutoConfig

c = AutoConfig.from_pretrained("unsloth/Qwen3-4B-Base")
V = c.vocab_size + 3059
H = c.hidden_size
print("Qwen3-4B: hidden %d, layers %d, vocab %d (incl. 3059 domain), tie=%s"
      % (H, c.num_hidden_layers, V, c.tie_word_embeddings))
emb = V * H
print("embedding/lm_head matrix: %.0fM params, %.1f%% of the model" % (emb / 1e6, 100 * emb / 4e9))

body = 2 * (4e9 - emb)          # forward FLOPs per token, everything but the head
head = 2 * H * V
print("\nper token, forward: body %.2f GFLOP | head %.2f GFLOP -> head is %.1f%%"
      % (body / 1e9, head / 1e9, 100 * head / (body + head)))
# the head also needs a weight gradient (tied embedding is trainable) and an input gradient
print("with the head's weight and input gradients: head is ~%.1f%% of forward+backward"
      % (100 * 3 * head / (2 * body + 3 * head)))

S, B = 800, 8
print("\nlogits at seq %d x batch %d: %.2f GB bf16, %.2f GB once cross-entropy upcasts to fp32"
      % (S, B, B * S * V * 2 / 1e9, B * S * V * 4 / 1e9))
print("positions the loss actually reads: 1-2 of %d (%.2f%%)" % (S, 100 * 2 / S))
