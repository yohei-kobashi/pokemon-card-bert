#!/usr/bin/env python3
"""Does merging the LoRA change which move the model picks?

Folding a LoRA into bf16 base weights is not bit-exact: W + BA is rounded once into bf16 instead
of being applied as a separate branch. The screen only reads the ARGMAX, so tiny logit shifts are
harmless -- unless they flip a near-tie.

This matters more than usual here. instance2's paired comparison is against `mirror_i2v40.json`,
produced by the UNMERGED path. If merging changes even a fraction of a percent of picks, part of
the next screen's movement is the scorer, not the checkpoint, and the loop's central statistic
silently stops measuring what it claims to.

Gate: report the agreement rate and the decisions that flip. Anything below ~99.9% means keep the
LoRA unmerged and take only the exact optimisations (logits_to_keep, KV reuse).
"""
import argparse
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
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()

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
    print("[data] %d decisions" % len(dec), flush=True)

    # merge=False so `before` really is the unmerged path; kv=False so the only difference
    # between the two passes is the weights.
    sc = QwenScorer(a.ckpt, merge=False, kv=False)
    before = [sc._score_card_first(*d) for d in dec]
    sc.model = sc.model.merge_and_unload()
    sc.model.eval()
    after = [sc._score_card_first(*d) for d in dec]

    agree = flips = 0
    gaps = []
    for k, (b, m) in enumerate(zip(before, after)):
        ib, im = max(range(len(b)), key=lambda i: b[i]), max(range(len(m)), key=lambda i: m[i])
        if ib == im:
            agree += 1
        else:
            flips += 1
            srt = sorted(b, reverse=True)
            gaps.append(srt[0] - srt[1])
            if flips <= 5:
                print("  flip %d: %d -> %d, top-2 logprob gap was %.4f"
                      % (k, ib, im, srt[0] - srt[1]), flush=True)
    n = len(dec)
    print("\nargmax agreement %d/%d = %.2f%%" % (agree, n, 100.0 * agree / n), flush=True)
    if gaps:
        print("flipped decisions had a mean top-2 gap of %.4f (near-ties flip; that is expected)"
              % (sum(gaps) / len(gaps)), flush=True)
    print("VERDICT: %s" % ("merge is safe" if agree / n >= 0.999 else
                           "DO NOT MERGE -- the scorer would confound the paired screen"),
          flush=True)


if __name__ == "__main__":
    main()
