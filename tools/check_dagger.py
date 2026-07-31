#!/usr/bin/env python3
"""Does the DAgger pool actually contain states the engine self-play pool does not?

The whole premise is distribution shift: the LM's mistakes create positions engine_v2 never
walks into, so training only on engine self-play leaves them uncovered. If the two pools turn
out to overlap heavily, DAgger adds nothing and the training run would be spent proving that
the hard way.

Checks, cheapest first:
  1. exact state-string overlap -- if a DAgger state already appears verbatim in the old pool,
     it is not new
  2. turn distribution -- distribution shift usually shows up as reaching different phases of
     the game, and the collapsed decks lose more when moving second
  3. label kind mix -- the new pool should be richer in exactly the decisions the LM gets
     wrong (play), not a copy of the old mix
"""
import collections
import gzip
import json
import re
import sys


def kind(s):
    m = re.match(r"([a-z_]+)", s or "")
    return m.group(1) if m else "?"


def load(path, limit, want_state=True):
    states, labels, turns = set(), collections.Counter(), collections.Counter()
    n = 0
    with gzip.open(path, "rt") as f:
        for line in f:
            d = json.loads(line)
            c = d.get("candidates") or []
            ch = d.get("chosen")
            if ch is None or ch >= len(c):
                continue
            n += 1
            if want_state:
                states.add(d["state"])
            labels[kind(c[ch])] += 1
            m = re.search(r" T(\d+)\.", d["state"])
            if m:
                turns[min(int(m.group(1)) // 3 * 3, 15)] += 1
            if limit and n >= limit:
                break
    return states, labels, turns, n


def main(dag_path, old_path, limit=200000):
    ds, dl, dt, dn = load(dag_path, limit)
    os_, ol, ot, on = load(old_path, limit)
    inter = len(ds & os_)
    print("dagger %d records (%d distinct states) | old %d records (%d distinct)"
          % (dn, len(ds), on, len(os_)))
    print("exact state overlap: %d = %.2f%% of dagger states"
          % (inter, 100.0 * inter / max(1, len(ds))))
    print("  -> %s" % ("MOSTLY THE SAME STATES: dagger adds little" if inter > 0.5 * len(ds)
                       else "the dagger pool is genuinely off the old distribution"))
    print("\nturn bucket      dagger    old")
    for t in sorted(set(dt) | set(ot)):
        print("  T%-3d %11.1f%% %6.1f%%" % (t, 100.0 * dt[t] / max(1, dn),
                                            100.0 * ot[t] / max(1, on)))
    print("\nlabel kind       dagger    old")
    for k in sorted(set(dl) | set(ol), key=lambda x: -(dl[x] + ol[x])):
        print("  %-12s %7.2f%% %6.2f%%" % (k, 100.0 * dl[k] / max(1, dn),
                                           100.0 * ol[k] / max(1, on)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2],
         int(sys.argv[3]) if len(sys.argv) > 3 else 200000)
