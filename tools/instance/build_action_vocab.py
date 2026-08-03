#!/usr/bin/env python3
"""Freeze the action-token vocabulary that a model will be trained and served with.

It is built from the OPTION side, not from the labels: at inference the model scores every legal
option, so an option whose token is missing cannot be picked at all, however often the engine
happened to choose it during training.

The list is written to a file and shipped alongside the checkpoint for the same reason
`domain_embeddings.pt` is -- a model whose vocabulary is rebuilt from different data is a model
whose every embedding row means something else. Rebuilding it is a new model, not a new run.

Measured on the v39+DAgger mix (1.2M decisions to build, a disjoint 300k to check):
    5,825 tokens | unknown options 0.007% | correct option unknown 0.010%
    every option unknown (must defer to engine_v2) 0.0003%
"""
import argparse
import collections
import gzip
import json
import re
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = the whole file")
    ap.add_argument("--min-count", type=int, default=1,
                    help="drop tokens seen fewer times; anything dropped becomes an option the "
                         "model can never pick, so raising this is a real cost")
    a = ap.parse_args()

    for p in ("/root/ptcg/repo", ".", "cg-lib"):
        if p not in sys.path:
            sys.path.insert(0, p)
    from lm.action_token import action_token

    RE = re.compile(r"(?:^| )(\d+)=(\S+)")
    vocab = collections.Counter()
    n = 0
    with gzip.open(a.data, "rt") as f:
        for line in f:
            d = json.loads(line)
            if not d.get("target"):
                continue
            n += 1
            for _, o in RE.findall(d["prompt"].rsplit(":: ", 1)[-1]):
                vocab[action_token(o)] += 1
            if a.limit and n >= a.limit:
                break

    keep = sorted(t for t, c in vocab.items() if c >= a.min_count)
    dropped = len(vocab) - len(keep)
    with open(a.out, "w") as f:
        json.dump({"tokens": keep, "built_from": a.data, "decisions": n,
                   "counts": {t: vocab[t] for t in keep}}, f)
    c = sorted(vocab[t] for t in keep)
    print("decisions %d | distinct option tokens %d | kept %d | dropped %d"
          % (n, len(vocab), len(keep), dropped), flush=True)
    print("  frequency: median %d | seen<5 %d (%.0f%%) | seen<20 %d (%.0f%%) | max %d"
          % (c[len(c) // 2], sum(1 for x in c if x < 5), 100.0 * sum(1 for x in c if x < 5) / len(c),
             sum(1 for x in c if x < 20), 100.0 * sum(1 for x in c if x < 20) / len(c), c[-1]),
          flush=True)
    print("-> %s" % a.out, flush=True)


if __name__ == "__main__":
    main()
