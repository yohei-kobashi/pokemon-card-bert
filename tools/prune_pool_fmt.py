#!/usr/bin/env python3
"""Keep only the rows written in one prompt format, and delete the rest.

The base pools are appended to every round and outlive several prompt formats. Training on a
mixture is worse than training on either: the model sees two input distributions for the same
task, and the screen -- which renders with ONE format -- then scores it on whichever the
checkpoint happened to be dominated by.

build_rerank stamps `pfmt` on every row it writes (rl_config.PROMPT_VERSIONS). Rows written
before the stamp existed have no `pfmt` key at all, which is precisely what identifies them as
the older format. So the filter is exact and needs no parsing of the prompt text.

    python3 tools/prune_pool_fmt.py --inp data/rerank/v40_base.jsonl.gz --count
    python3 tools/prune_pool_fmt.py --inp data/rerank/v40_base.jsonl.gz --keep v41 --apply

`--count` reports the composition and exits -- run it first, and only prune once the surviving
count is large enough to train a round on. Without `--apply` the pruned file is written next to
the input and the original is left alone; with it, the swap is an atomic `mv`, the same
discipline pool_daemon uses so a run already reading the pool keeps its inode.
"""

import argparse
import collections
import gzip
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inp", required=True)
    ap.add_argument("--keep", default="v41", help="pfmt value to KEEP")
    ap.add_argument("--out", default="", help="default: <inp>.pruned.jsonl.gz")
    ap.add_argument("--count", action="store_true", help="report composition and exit")
    ap.add_argument("--apply", action="store_true",
                    help="replace the input with the pruned file (atomic mv)")
    ap.add_argument("--min-keep", type=int, default=1_000_000,
                    help="refuse to --apply if fewer rows would survive; a pool pruned below "
                         "what a round draws starves the next training run")
    a = ap.parse_args()

    if not os.path.exists(a.inp):
        sys.exit("missing: %s" % a.inp)

    if a.count:
        c = collections.Counter()
        n = 0
        for line in gzip.open(a.inp, "rt"):
            try:
                c[json.loads(line).get("pfmt") or "(unstamped)"] += 1
            except Exception:
                c["(unparseable)"] += 1
            n += 1
            if n % 2_000_000 == 0:
                print("  %d rows..." % n, flush=True)
        print("%s: %d rows" % (a.inp, n))
        for k, v in c.most_common():
            print("  %-14s %12d  %5.1f%%" % (k, v, 100 * v / max(1, n)))
        return

    out = a.out or (a.inp[:-len(".jsonl.gz")] + ".pruned.jsonl.gz"
                    if a.inp.endswith(".jsonl.gz") else a.inp + ".pruned")
    tmp = out + ".part"
    n = kept = 0
    with gzip.open(a.inp, "rt") as f, gzip.open(tmp, "wt") as g:
        for line in f:
            n += 1
            try:
                if json.loads(line).get("pfmt") != a.keep:
                    continue
            except Exception:
                continue
            g.write(line)
            kept += 1
            if n % 2_000_000 == 0:
                print("  %d rows, %d kept..." % (n, kept), flush=True)
    print("%d rows -> %d kept (%.1f%%) as %s" % (n, kept, 100 * kept / max(1, n), tmp))

    if not a.apply:
        os.replace(tmp, out)
        print("wrote %s (input untouched; re-run with --apply to swap)" % out)
        return
    if kept < a.min_keep:
        os.remove(tmp)
        sys.exit("REFUSING: only %d rows carry pfmt=%s, below --min-keep %d. Let the pool grow."
                 % (kept, a.keep, a.min_keep))
    # Verify before swapping: a truncated .gz decompresses fine up to the cut, so a short write
    # would replace the pool with a partial one and nothing would error.
    check = 0
    for line in gzip.open(tmp, "rt"):
        json.loads(line)
        check += 1
    if check != kept:
        os.remove(tmp)
        sys.exit("REFUSING: re-read got %d rows, wrote %d" % (check, kept))
    os.replace(tmp, a.inp)
    print("pool replaced: %s now holds %d rows, all pfmt=%s" % (a.inp, kept, a.keep))


if __name__ == "__main__":
    main()
