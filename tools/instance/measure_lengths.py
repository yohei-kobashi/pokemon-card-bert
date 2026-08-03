#!/usr/bin/env python3
"""Measure the SFT prompt-length distribution under a given tokenizer, and price the padding.

Everything about SFT throughput here is decided by this histogram, so it is measured before any
config is chosen rather than after a slow run:

  * PADDING. Each batch is padded to its own longest member. Under random order that is close to
    the global maximum every time, so the fraction of compute spent on pad tokens is
    1 - mean/max. Sorting by length (group_by_length) collapses the per-batch max toward the
    per-batch mean; this script simulates both orders at the real batch size and reports the
    difference as a speed ceiling.
  * MAXLEN. A prompt longer than max_length is truncated, and for this task truncation is not a
    quality tax but a correctness bug: the answer slot is the LAST position, and the menu the
    answer indexes sits at the very end of the prompt. Right-truncation deletes exactly the part
    the model must read -- the failure already recorded in `rerank-prompt-truncation-bug`, where
    it silently destroyed 99% of decisions. The over-length share is therefore reported
    separately and loudly.
"""
import argparse
import gzip
import json
import random
import statistics
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="unsloth/Qwen3-4B-Base")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--bsz", type=int, default=8)
    ap.add_argument("--maxlen", type=int, default=1024)
    ap.add_argument("--domain-tokens", action="store_true")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(a.model)
    tk = getattr(tk, "tokenizer", tk)
    if a.domain_tokens:
        for p in ("/root/ptcg/repo", "/root/ptcg/repo/cg-lib", ".", "cg-lib"):
            if p not in sys.path:
                sys.path.insert(0, p)
        from lm.vocab import special_tokens
        n = tk.add_tokens(special_tokens())
        print("[tok] added %d domain tokens -> vocab %d" % (n, len(tk)), flush=True)

    lens = []
    with gzip.open(a.data, "rt") as f:
        for line in f:
            d = json.loads(line)
            p = d.get("prompt")
            if not p:
                continue
            # +1 for the answer token the completion contributes
            lens.append(len(tk(p, add_special_tokens=False)["input_ids"]) + 1)
            if len(lens) >= a.n:
                break
    lens.sort()
    q = lambda p: lens[min(len(lens) - 1, int(p * len(lens)))]
    print("\n[len] n=%d  mean %.0f  median %d  p90 %d  p99 %d  p99.9 %d  max %d"
          % (len(lens), statistics.mean(lens), q(.5), q(.9), q(.99), q(.999), lens[-1]))

    over = sum(1 for x in lens if x > a.maxlen)
    print("[len] over maxlen=%d: %d (%.2f%%)  %s"
          % (a.maxlen, over, 100.0 * over / len(lens),
             "<- these lose their menu AND answer slot to right-truncation" if over else "none"))

    def cost(order, bsz):
        """padded tokens per batch, summed -- the compute actually spent"""
        tot = 0
        for i in range(0, len(order), bsz):
            b = order[i:i + bsz]
            tot += max(b) * len(b)
        return tot

    real = sum(lens)
    rnd = list(lens)
    random.Random(0).shuffle(rnd)
    for bsz in sorted({a.bsz, 8, 16, 32}):
        cr = cost(rnd, bsz)
        cs = cost(sorted(lens), bsz)
        print("[pad] bsz %2d | random order %.2fx the useful tokens | length-sorted %.2fx"
              "  -> group_by_length ceiling %+.0f%% throughput"
              % (bsz, cr / real, cs / real, 100.0 * (cr / cs - 1)))
    print("\n[pad] 'useful tokens' = sum of real lengths. A 1.60x means 60%% of the forward pass "
          "is spent on padding.")


if __name__ == "__main__":
    main()
