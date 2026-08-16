#!/usr/bin/env python3
"""How many attach candidates are indistinguishable IN THE RENDERED PROMPT?

`attach_ties.py` answered this against the observation. This asks the stricter and more useful
question: two targets the MODEL cannot tell apart. The model sees only the state string, so if
two attach candidates point at board slots whose rendered descriptors are byte-identical, no
amount of training can make it prefer engine_v2's arbitrary pick between them -- the label is
noise and the top1 ceiling sits below 100%.

Uses the DAgger pool because those are the states the model's own play reaches.
"""
import collections
import gzip
import json
import re
import sys

ME = re.compile(r"\bME (.*?)(?: \| OP |$)", re.S)
ACT = re.compile(r"\bA\[(.*?)\]")
BEN = re.compile(r"\bB\[(.*?)\]")


def split_entries(s):
    """bench entries are comma separated; card descriptors have no commas in them"""
    return [x.strip() for x in s.split(",") if x.strip()]


def slots(state):
    m = ME.search(state)
    if not m:
        return {}
    seg = m.group(1)
    out = {}
    a = ACT.search(seg)
    if a:
        for i, e in enumerate(split_entries(a.group(1))):
            out["ACTIVE%d" % i] = e
    b = BEN.search(seg)
    if b:
        for i, e in enumerate(split_entries(b.group(1))):
            out["BENCH%d" % i] = e
    return out


def main(path):
    n_dec = n_tied_dec = 0
    cands = distinct = 0
    grp_of_label = collections.Counter()
    unresolved = 0
    examples = []
    for line in gzip.open(path, "rt"):
        d = json.loads(line)
        c = d["candidates"]
        at = [i for i, x in enumerate(c) if x.startswith("attach:")]
        if len(at) < 2:
            continue
        sl = slots(d["state"])
        keys, miss = [], False
        for i in at:
            tgt = c[i].split("@", 1)[1].split("#")[0] if "@" in c[i] else ""
            desc = sl.get(tgt)
            if desc is None:
                miss = True
                break
            # the ENERGY being attached matters too: attach:c1@X vs attach:c5@X differ
            keys.append((c[i].split(":", 1)[1].split("@", 1)[0], desc))
        if miss:
            unresolved += 1
            continue
        n_dec += 1
        cands += len(at)
        g = collections.Counter(keys)
        distinct += len(g)
        if len(g) < len(at):
            n_tied_dec += 1
        if d["chosen"] in at:
            grp_of_label[g[keys[at.index(d["chosen"])]]] += 1
            if g[keys[at.index(d["chosen"])]] >= 3 and len(examples) < 2:
                examples.append((d["state"][-420:], [c[i] for i in at], c[d["chosen"]]))

    print("attach decisions with >=2 attach candidates : %d   (%d unparseable, skipped)"
          % (n_dec, unresolved))
    print("attach candidates per decision              : %.2f" % (cands / max(1, n_dec)))
    print("distinguishable in the prompt               : %.2f  (%.1f%%)"
          % (distinct / max(1, n_dec), 100.0 * distinct / max(1, cands)))
    print("decisions holding indistinguishable ones    : %.1f%%"
          % (100.0 * n_tied_dec / max(1, n_dec)))
    tot = sum(grp_of_label.values())
    ceil = sum(c / g for g, c in grp_of_label.items()) / max(1, tot)
    print("tied-group size the LABEL landed in         : %s"
          % ", ".join("%d:%d" % (g, c) for g, c in sorted(grp_of_label.items())))
    print("TOP1 CEILING for a perfect model            : %.1f%%" % (100.0 * ceil))
    for st, cs, lab in examples:
        print("\n--- a decision whose label sits in a >=3 tie ---\n%s" % st)
        print("attach candidates: %s\nengine chose: %s" % (cs, lab))


if __name__ == "__main__":
    main(sys.argv[1])
