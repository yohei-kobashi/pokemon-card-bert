#!/usr/bin/env python3
"""Time the patched QwenScorer as it will actually run, with and without the LoRA merge.

The merge is the largest single win (1.88x) and the only one that is not numerically exact:
tools/check_merge_equiv.py measured 99.00% argmax agreement over 400 real decisions, with every
flip on a near-tie (mean top-2 logprob gap 0.0069, three of four below 0.0001).

That 1% matters here for a specific reason, not a general one. instance2's paired screen compares
a new checkpoint against `mirror_i2v40.json`, which was produced by the unmerged scorer -- so
merging would put part of the difference in the instrument. Collection has no such constraint:
collect_dagger's labels come from engine_v2 and the LM only steers which states are visited.

So the question this answers is: what do the EXACT optimisations alone buy, and is the merge
worth its comparability cost on top?
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--no-kv", action="store_true")
    ap.add_argument("--no-fast", action="store_true", help="the pre-optimisation path")
    ap.add_argument("--backend", default="hf", choices=["hf", "unsloth"])
    a = ap.parse_args()

    import torch
    from tools.mirror_match import QwenScorer

    dec = []
    with gzip.open(a.data, "rt") as f:
        for line in f:
            d = json.loads(line)
            c = [t for _, t in _OPT.findall(d["prompt"].rsplit(":: ", 1)[-1])]
            if len(c) >= 2:
                dec.append((d["prompt"], c))
                if len(dec) >= a.n:
                    break

    sc = QwenScorer(a.ckpt, merge=a.merge, kv=not (a.no_kv or a.no_fast),
                    backend=a.backend)
    if a.no_fast:
        sc._klast = {}
    for i in range(3):
        sc._score_card_first(*dec[i])
    torch.cuda.synchronize()
    t0 = time.time()
    for d in dec:
        sc._score_card_first(*d)
    torch.cuda.synchronize()
    el = (time.time() - t0) / len(dec)
    tag = "%s merge=%s kv=%s ltk=%s" % (a.backend, a.merge, sc.kv, bool(sc._klast))
    print("\nRESULT  %-46s %6.1f ms/decision   %.2fx vs 134.1 ms baseline"
          % (tag, 1000 * el, 0.1341 / el), flush=True)


if __name__ == "__main__":
    main()
