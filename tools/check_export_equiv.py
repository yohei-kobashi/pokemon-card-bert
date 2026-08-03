#!/usr/bin/env python3
"""Is the exported standalone model the same model QwenScorer assembles at load time?

`export_qwen_merged.py` rebuilds by hand what the scorer builds in memory: resize the embedding,
restore 3,064 domain rows, attach the LoRA, merge. Any step done differently -- rows written to
the wrong offset, the tokenizer taken from the base instead of the checkpoint, the LoRA silently
not found -- produces a model that loads, runs, and is wrong. vLLM would then be benchmarked on
a different model than the one it is being compared against, and the number would be meaningless
in a way no error message would reveal.

Both are scored through the SAME `_score_card_first`, so the only variable is the weights.
"""
import argparse
import gc
import gzip
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

_OPT = re.compile(r"(?:^| )(\d+)=(\S+)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="the adapter checkpoint")
    ap.add_argument("--export", required=True, help="the standalone directory")
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=200)
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

    sc = QwenScorer(a.ckpt, backend="hf", merge=True, kv=True)
    ref = [sc._score_card_first(*d) for d in dec]
    del sc
    gc.collect()
    torch.cuda.empty_cache()

    # The export has no adapter_config.json, so QwenScorer loads it as a plain base model with
    # nothing to merge -- which is exactly what it is.
    sc2 = QwenScorer(a.export, base=a.export, backend="hf", merge=True, kv=True)
    got = [sc2._score_card_first(*d) for d in dec]

    worst = max(abs(x - y) for r, g in zip(ref, got) for x, y in zip(r, g))
    agree = sum(1 for r, g in zip(ref, got)
                if max(range(len(r)), key=lambda i: r[i])
                == max(range(len(g)), key=lambda i: g[i]))
    print("\nexport vs in-memory: argmax %d/%d = %.2f%% | max |diff| %.2e"
          % (agree, len(dec), 100.0 * agree / len(dec), worst), flush=True)
    ok = agree == len(dec) and worst < 1e-2
    print("VERDICT: %s" % ("the export IS the same model" if ok else
                           "MISMATCH -- do not benchmark against this export"), flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
