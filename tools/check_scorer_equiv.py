#!/usr/bin/env python3
"""Does the optimised scorer pick the same moves as the one the baselines were made with?

None of the three optimisations is bit-exact, and pretending otherwise would be the mistake:

    logits_to_keep=1   computing logits at one position instead of 368 changes the reduction
                       order -- measured 0.0625 max on raw logits
    KV reuse           attention over a cache is a different kernel path -- 0.125 max on
                       log-probs, same argmax
    merged LoRA        W + BA is rounded once into bf16 instead of applied as a branch

The screen reads only the ARGMAX, and the paired statistic it produces has its own run-to-run
noise (the same checkpoint has re-scored 2.6pt apart on a comparable protocol). So the question
is not whether the numbers move but whether the PICKS do, and by how much on decisions the model
was not already indifferent about.

Both scorers are loaded in the same process on the same decisions, the reference first, so the
comparison is paired and nothing about the data can differ between them.
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


def score_all(dec, **kw):
    import torch
    from tools.mirror_match import QwenScorer
    sc = QwenScorer(kw.pop("ckpt"), **kw)
    out = [sc._score_card_first(*d) for d in dec]
    del sc
    gc.collect()
    torch.cuda.empty_cache()
    return out


def run_child(args, out_path):
    """Score in a SEPARATE PROCESS.

    `import unsloth` patches transformers globally and does not undo it, so loading the unsloth
    reference first silently disables logits_to_keep and the KV cache in the hf model loaded
    after it -- the first attempt at this comparison reported the reference with the fast paths
    ON and the optimised scorer with them OFF, which is exactly backwards. One process per
    scorer is the only way the flags mean what they say.
    """
    import subprocess
    cmd = [sys.executable, os.path.abspath(__file__), "--child", out_path] + args
    subprocess.run(cmd, check=True, cwd=ROOT)
    return json.load(open(out_path))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--child", default="", help=argparse.SUPPRESS)
    ap.add_argument("--backend", default="hf")
    ap.add_argument("--merge", type=int, default=1)
    ap.add_argument("--kv", type=int, default=1)
    a = ap.parse_args()

    dec = []
    with gzip.open(a.data, "rt") as f:
        for line in f:
            d = json.loads(line)
            c = [t for _, t in _OPT.findall(d["prompt"].rsplit(":: ", 1)[-1])]
            if len(c) >= 2:
                dec.append((d["prompt"], c))
                if len(dec) >= a.n:
                    break
    if a.child:
        out = score_all(dec, ckpt=a.ckpt, backend=a.backend,
                        merge=bool(a.merge), kv=bool(a.kv))
        json.dump(out, open(a.child, "w"))
        return
    print("[data] %d decisions" % len(dec), flush=True)

    common = ["--ckpt", a.ckpt, "--data", a.data, "--n", str(a.n)]
    print("\n=== reference: the path every existing baseline was made with ===", flush=True)
    ref = run_child(common + ["--backend", "unsloth", "--merge", "0", "--kv", "0"],
                    "/tmp/scorer_ref.json")
    print("\n=== optimised: hf backend, merged LoRA, logits_to_keep, KV reuse ===", flush=True)
    new = run_child(common + ["--backend", "hf", "--merge", "1", "--kv", "1"],
                    "/tmp/scorer_new.json")

    agree = 0
    gaps, big = [], 0
    for k, (b, m) in enumerate(zip(ref, new)):
        ib = max(range(len(b)), key=lambda i: b[i])
        im = max(range(len(m)), key=lambda i: m[i])
        if ib == im:
            agree += 1
            continue
        srt = sorted(b, reverse=True)
        g = srt[0] - srt[1]
        gaps.append(g)
        if g > 0.05:
            big += 1
        if len(gaps) <= 6:
            print("  flip %d: %d -> %d, top-2 logprob gap %.5f" % (k, ib, im, g), flush=True)
    n = len(dec)
    print("\nargmax agreement %d/%d = %.2f%%" % (agree, n, 100.0 * agree / n), flush=True)
    if gaps:
        gaps.sort()
        print("flips: %d | median top-2 gap %.5f | max %.5f | %d with a gap > 0.05"
              % (len(gaps), gaps[len(gaps) // 2], gaps[-1], big), flush=True)
        print("A flip on a gap of ~0 is a decision the model rated as a tie; the coin landed the "
              "other way. Only the %d flips above 0.05 are the model actually changing its mind "
              "(%.2f%% of decisions)." % (big, 100.0 * big / n), flush=True)


if __name__ == "__main__":
    main()
