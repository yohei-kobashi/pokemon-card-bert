#!/usr/bin/env python3
"""Round 2: the levers that survive the round-1 measurement.

Round 1 on the RTX 5880 Ada, 120 real decisions, prompts mean 368 / p90 460 tokens:

    baseline (PeftModel, 2nd token re-prefills)   134.1 ms/decision
    + merged LoRA                                  71.3 ms/decision   1.88x
    first forward alone, batch 1                   45.2 ms/decision
    first forward alone, batch 4                   43.2 ms/decision   <- 1.05x over batch 1
    first forward alone, batch 32                  53.6 ms/decision   <- WORSE

BATCHING IS DEAD. A 368-token prefill of a 4B model is ~2.9 TFLOP; at 45 ms that is ~65 TFLOP/s,
which is most of what this card does in bf16. There is no idle GPU for a bigger batch to fill,
which also removes vLLM's main lever -- continuous batching and paged attention optimise
memory-bound decode and scheduling, and this workload is neither.

What is left is to do less work per decision:

  logits_to_keep   a plain forward computes logits at EVERY position. Only the last one is read.
                   That is 368 x 251,048 x 2560 x 2 = 473 GFLOP of lm_head (16% on top of the
                   2.9 TFLOP body) and a 368 x 251,048 tensor materialised and upcast to fp32
                   -- 740 MB of pure memory traffic per forward.
  kv               the tie-break forward re-runs the whole prompt to score ONE extra position.
                   Round 1's attempt broke on transformers 5.x (past_key_values is a Cache
                   object, not a tuple); DynamicCache.crop puts the cache back afterwards so the
                   same prompt cache serves every group.
  compile          the gap between 71.3 ms measured and ~50 ms of GPU work is Python and kernel
                   launch overhead. reduce-overhead mode captures CUDA graphs, which is the same
                   thing vLLM would buy us here -- without touching the environment the training
                   runs in.
"""
import argparse
import gzip
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

_OPT = re.compile(r"(?:^| )(\d+)=(\S+)")


def load_decisions(path, n, minc=2):
    out = []
    with gzip.open(path, "rt") as f:
        for line in f:
            d = json.loads(line)
            pr = d["prompt"]
            c = [t for _, t in _OPT.findall(pr.rsplit(":: ", 1)[-1])]
            if len(c) >= minc:
                out.append((pr, c))
                if len(out) >= n:
                    break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--maxlen", type=int, default=1024)
    ap.add_argument("--compile", action="store_true")
    a = ap.parse_args()

    import torch
    from tools.mirror_match import QwenScorer
    from lm.action_token import first_token, second_token, equivalent, to_scheme_b

    dec = load_decisions(a.data, a.n)
    sc = QwenScorer(a.ckpt, maxlen=a.maxlen)
    tk = sc.tk
    sc.model = sc.model.merge_and_unload()
    sc.model.eval()
    model = sc.model
    dev = model.device
    print("\n[all timings below are WITH the LoRA already merged]", flush=True)

    enc = {}

    def ids_of(prompt):
        if prompt not in enc:
            p = to_scheme_b(prompt) if sc.scheme_b else prompt
            enc[prompt] = tk(p, add_special_tokens=False, truncation=True,
                             max_length=a.maxlen)["input_ids"]
        return enc[prompt]

    def groups_needing_tiebreak(cands, heads):
        need = {}
        for h in set(heads):
            g = [i for i, x in enumerate(heads) if x == h]
            if len(g) > 1 and not all(equivalent(cands[i], cands[g[0]]) for i in g):
                need[h] = g
        return need

    def make(keep_logits, use_kv):
        def f(d):
            prompt, cands = d
            ids = ids_of(prompt)
            heads = [first_token(c) for c in cands]
            t = torch.tensor([ids], device=dev)
            kw = {"logits_to_keep": 1} if keep_logits else {}
            with torch.no_grad():
                if use_kv:
                    from transformers import DynamicCache
                    cache = DynamicCache()
                    out = model(input_ids=t, past_key_values=cache, use_cache=True, **kw)
                    cache = out.past_key_values
                else:
                    out = model(input_ids=t, use_cache=False, **kw)
                lp1 = torch.log_softmax(out.logits[0, -1, :].float(), -1)
            hid = {h: tk.convert_tokens_to_ids(h) for h in set(heads)}
            res = [float(lp1[hid[h]]) if hid[h] is not None else -1e9 for h in heads]
            for h, grp in groups_needing_tiebreak(cands, heads).items():
                with torch.no_grad():
                    if use_kv:
                        o2 = model(input_ids=torch.tensor([[hid[h]]], device=dev),
                                   past_key_values=cache, use_cache=True, **kw)
                        cache.crop(len(ids))          # put it back for the next group
                    else:
                        o2 = model(input_ids=torch.tensor([ids + [hid[h]]], device=dev),
                                   use_cache=False, **kw)
                    lp2 = torch.log_softmax(o2.logits[0, -1, :].float(), -1)
                for i in grp:
                    t2 = second_token(cands[i]) if sc.scheme_b else None
                    j = tk.convert_tokens_to_ids(t2) if t2 else None
                    res[i] += float(lp2[j]) if j is not None else -1e9
            return res
        return f

    ref_fn = make(False, False)
    ref = [ref_fn(d) for d in dec[:8]]

    def run(fn, label):
        try:
            got = [fn(d) for d in dec[:8]]
            worst = max(abs(x - y) for r, g in zip(ref, got) for x, y in zip(r, g))
        except Exception as e:
            print("  %-34s FAILED: %s" % (label, str(e)[:110]), flush=True)
            return None
        for i in range(3):
            fn(dec[i])
        torch.cuda.synchronize(); t0 = time.time()
        for d in dec:
            fn(d)
        torch.cuda.synchronize()
        el = (time.time() - t0) / len(dec)
        print("  %-34s %6.1f ms/decision   %.2fx vs merged   max|diff| %.2e"
              % (label, 1000 * el, BASE / el, worst), flush=True)
        return el

    print("\n[merged, no other change]", flush=True)
    torch.cuda.synchronize(); t0 = time.time()
    for d in dec:
        ref_fn(d)
    torch.cuda.synchronize()
    BASE = (time.time() - t0) / len(dec)
    print("  %-34s %6.1f ms/decision   1.00x" % ("merged", 1000 * BASE), flush=True)

    run(make(True, False), "+ logits_to_keep=1")
    run(make(False, True), "+ kv cache")
    best = run(make(True, True), "+ logits_to_keep + kv cache")

    if a.compile:
        print("\n[+ torch.compile reduce-overhead]", flush=True)
        try:
            model.forward = torch.compile(model.forward, mode="reduce-overhead",
                                          dynamic=True, fullgraph=False)
            run(make(True, True), "compiled + logits + kv")
        except Exception as e:
            print("  compile FAILED: %s" % str(e)[:140], flush=True)

    print("\npeak VRAM %.1f GiB" % (torch.cuda.max_memory_allocated() / 2**30), flush=True)
    if best:
        print("\nProjection for one 63-deck screen shard (62,002 decisions):", flush=True)
        print("  today  0.199 s/decision in-screen -> 3.43 h of the 5.5 h shard", flush=True)
        print("  now    %.3f s standalone vs %.3f s standalone baseline = %.2fx GPU work"
              % (best, 0.1341, 0.1341 / best), flush=True)


if __name__ == "__main__":
    main()
