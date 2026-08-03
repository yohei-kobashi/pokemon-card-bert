#!/usr/bin/env python3
"""Freeze the first-token set for the card-first answer scheme.

Most of it needs no new rows at all: a card token like `c1227` and an attack token like `a368`
are already in lm.vocab.special_tokens(), already appear in every prompt, and -- because Qwen3-4B
ties its embedding to its output head -- their output rows are the rows the prompt occurrences
are already training. What is genuinely new is only the handful of options that name no card
(`A|end`, `A|retreat`, `A|num|3`, ...) plus the <sN> tie-breakers.

That is the whole point of the scheme, so the split is REPORTED rather than assumed: if the
"needs a new row" count came back in the thousands, the tail this was meant to remove would
simply have moved.
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
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    for p in ("/root/ptcg/repo", ".", "cg-lib"):
        if p not in sys.path:
            sys.path.insert(0, p)
    from lm.action_token import first_token, sub_index, MAX_SUB, SUB_TOKENS
    from lm.vocab import special_tokens

    known = set(special_tokens())
    RE = re.compile(r"(?:^| )(\d+)=(\S+)")
    firsts = collections.Counter()
    subs = collections.Counter()
    n = two = 0
    with gzip.open(a.data, "rt") as f:
        for line in f:
            d = json.loads(line)
            t = d.get("target")
            if not t:
                continue
            opts = [o for _, o in RE.findall(d["prompt"].rsplit(":: ", 1)[-1])]
            k = int(t)
            if k >= len(opts):
                continue
            n += 1
            for o in opts:
                firsts[first_token(o)] += 1
            s = sub_index(d["prompt"], opts, k)
            if s is not None:
                two += 1
                subs[s] += 1
            if a.limit and n >= a.limit:
                break

    new = sorted(t for t in firsts if t not in known)
    old = [t for t in firsts if t in known]
    max_sub = max(subs) if subs else 0
    json.dump({"first_tokens": sorted(firsts), "new_tokens": new,
               "sub_tokens": SUB_TOKENS, "max_sub_used": max_sub,
               "decisions": n, "counts": {t: firsts[t] for t in firsts}},
              open(a.out, "w"))
    print("decisions %d" % n, flush=True)
    print("first-token vocabulary %d = %d already in the domain vocabulary + %d new"
          % (len(firsts), len(old), len(new)), flush=True)
    print("  new ones: %s" % ", ".join(new[:24]) + (" ..." if len(new) > 24 else ""), flush=True)
    print("second token needed on %d decisions (%.2f%%) -> %.3f forwards each"
          % (two, 100.0 * two / n, 1 + two / n), flush=True)
    print("  highest <sN> used: %d (alphabet is %d)" % (max_sub, MAX_SUB), flush=True)
    if subs:
        tot = sum(subs.values())
        print("  <s0> %.1f%%  <s1> %.1f%%  <s2> %.1f%%"
              % (100.0 * subs[0] / tot, 100.0 * subs[1] / tot, 100.0 * subs[2] / tot), flush=True)
    print("-> %s" % a.out, flush=True)


if __name__ == "__main__":
    main()
