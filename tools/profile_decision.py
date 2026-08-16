#!/usr/bin/env python3
"""Where does a 4B decision's time actually go -- GPU, or Python?

    PYTHONPATH=cg-lib:tools:. python3 tools/profile_decision.py --adapter /root/out/lora_x

The batch sweep (tools/bench_prefill.py) says the FORWARD is saturated at batch 1: an 800-token
prefill already reaches ~73% of a pure large GEMM's throughput on this card, so there is no idle
GPU for a bigger batch to fill. That answers "why doesn't batching help" but not "why is a
decision 134 ms", and those are different questions -- mirror_match's 134 ms is per DECISION,
while the sweep measures one forward. The gap is tokenisation, the card-first grouping, the
tie-break pass and the Python around them.

That distinction decides what to build. If the gap is large, the server's throughput is bound by
CPU on a box with 13.44 effective cores, and the fix is to stop serialising the CPU part behind
the GPU lock -- not to batch.
"""
import argparse
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "tools", "instance")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--tokens", type=int, default=368, help="mirror_match measured 134 ms here")
    ap.add_argument("--cands", type=int, default=6)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()

    import torch
    from mirror_match import QwenScorer
    sc = QwenScorer(a.adapter, merge=a.merge, maxlen=1024)

    # A prompt of the right SHAPE: the card-first path reads the menu out of the text, so the
    # candidates must be real encodings rather than "opt0".
    filler = " ".join("c%d" % (i % 900 + 1) for i in range(a.tokens))
    cands = ["play:c%d@BENCH%d" % (100 + i, i) for i in range(a.cands)]
    prompt = ("[ACT]\nDECK win[%s] || SEL MAIN n1-1 :: %s"
              % (filler, " ".join("%d=%s" % (i, c) for i, c in enumerate(cands))))
    ntok = len(sc.tk(prompt, add_special_tokens=False)["input_ids"])
    print("prompt %d tokens, %d candidates, weights %s"
          % (ntok, len(cands), "merged" if a.merge else "lora (as the server runs it)"))

    for _ in range(3):
        sc.score(prompt, cands)
    torch.cuda.synchronize()

    tot, tok_t, fwd_t = [], [], []
    ids = None
    for _ in range(a.n):
        t0 = time.time()
        sc.score(prompt, cands)
        torch.cuda.synchronize()
        tot.append(time.time() - t0)

        t1 = time.time()                                  # tokenisation alone (pure CPU)
        ids = sc.tk(prompt, add_special_tokens=False, truncation=True,
                    max_length=sc.maxlen)["input_ids"]
        tok_t.append(time.time() - t1)

        t2 = time.time()                                  # one forward alone (GPU)
        with torch.no_grad():
            sc.model(input_ids=torch.tensor([ids], device=sc.model.device),
                     use_cache=False, **sc._klast)
        torch.cuda.synchronize()
        fwd_t.append(time.time() - t2)

    m = lambda x: 1000 * statistics.median(x)             # noqa: E731
    d, t, f = m(tot), m(tok_t), m(fwd_t)
    print("\n  full decision      %7.1f ms" % d)
    print("  one forward (GPU)  %7.1f ms   %4.0f%% of the decision" % (f, 100 * f / d))
    print("  tokenise (CPU)     %7.1f ms   %4.0f%%" % (t, 100 * t / d))
    print("  everything else    %7.1f ms   %4.0f%%   <- Python: trie/grouping/gather/tie-break"
          % (d - f - t, 100 * (d - f - t) / d))
    print("\n  throughput if the GPU lock covered ONLY the forward: %.1f decisions/s"
          % (1000.0 / f))
    print("  throughput as the server serialises it today:          %.1f decisions/s"
          % (1000.0 / d))


if __name__ == "__main__":
    main()
