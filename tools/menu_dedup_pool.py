#!/usr/bin/env python3
"""Rewrite an already-built pool's SEL menu to one entry per ACT, without replaying the games.

The menu is recoverable FROM THE MENU: every entry is a rendered option string, and the board
the duplicate test needs is in the same prompt. So the whole prompt-format change is a text
rewrite -- `to_scheme_b` already relies on the same property. Replaying 2.9M decisions to
re-serialise them would cost days and would not produce the identical states anyway, because the
engine RNG is not seedable.

What changes, measured over 60,000 base-pool decisions:

    menu entries per decision   7.08 -> 5.36   (-24.4%)
    menu characters             111  -> 79     (-29.4%)
    whole prompt                                (-4.8%)

`candidates` and `chosen` are NOT touched: they were already deduped by the same rule, so this
only makes the prompt agree with what the model is asked to rank.

`menu_index` IS invalidated -- entries are renumbered over the surviving acts. It is dropped
rather than silently left wrong, so a decoder pool built from this file fails loudly instead of
training on targets that point at the wrong option.
"""
import argparse
import gzip
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib")):
    if p not in sys.path:
        sys.path.insert(0, p)

from lm.action_token import dedup_options       # noqa: E402

_MENU = re.compile(r"(?:^| )(\d+)=(\S+)")
STOP = "stop"


def rewrite(state):
    """-> (new state, entries before, entries after). Unchanged if the menu cannot be read."""
    head, sep, menu = state.rpartition(" :: ")
    if not sep:
        return state, 0, 0
    texts = [t for _n, t in _MENU.findall(menu)]
    if len(texts) < 2:
        return state, len(texts), len(texts)
    # a trailing STOP is a pseudo-option, not a game act; keep it last and out of the dedup
    stop = texts and texts[-1] == STOP
    body = texts[:-1] if stop else texts
    keep = dedup_options(body, state=state)[0]
    if stop:
        keep = keep + [STOP]
    return (head + " :: " + " ".join("%d=%s" % (i, t) for i, t in enumerate(keep)),
            len(texts), len(keep))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    n = before = after = clen = nlen = 0
    dropped_idx = 0
    with gzip.open(a.inp, "rt") as f, gzip.open(a.out, "wt") as g:
        for line in f:
            d = json.loads(line)
            s = d.get("state")
            if not s:
                g.write(line)
                continue
            ns, b, af = rewrite(s)
            n += 1
            before += b
            after += af
            clen += len(s)
            nlen += len(ns)
            d["state"] = ns
            if d.pop("menu_index", None) is not None:
                dropped_idx += 1
            g.write(json.dumps(d, ensure_ascii=False) + "\n")
    print("%s -> %s" % (a.inp, a.out))
    print("  decisions %d | menu entries %.2f -> %.2f (-%.1f%%)"
          % (n, before / max(1, n), after / max(1, n),
             100.0 * (before - after) / max(1, before)))
    print("  prompt chars %.0f -> %.0f (-%.1f%%)"
          % (clen / max(1, n), nlen / max(1, n), 100.0 * (clen - nlen) / max(1, clen)))
    print("  menu_index dropped on %d records (renumbering invalidates it)" % dropped_idx)


if __name__ == "__main__":
    main()
