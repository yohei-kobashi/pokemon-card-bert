#!/usr/bin/env python3
"""Step 0: does menu index k in the rendered prompt correspond to cands[k]?

Everything downstream assumes it.  lm/serialize.py:183 builds the menu as
`f"{i}={encode_option(o, obs)}"` over `sel["option"]`, and rl_rollout:117 builds
`cands = [encode_option(o, obs) for o in opts]` -- the same function over the same list, so they
should agree by construction.  But rl_rollout's MULTIPICK path (line 131) scores only the
`remaining` options and appends STOP, which is a different, shorter list.  If those records reach
the branch-point set, index k means two different things and every number built on top is wrong.

So this parses the menu back out of each prompt and compares it to `cands` element by element.
"""
import glob
import gzip
import json
import re
import sys
from collections import Counter

MENU_RE = re.compile(r"(?:^| )(\d+)=")


def parse_menu(prompt):
    """-> list of option strings, indexed as the model sees them."""
    tail = prompt.rsplit(":: ", 1)[-1]
    hits = list(MENU_RE.finditer(tail))
    out = []
    for j, m in enumerate(hits):
        if int(m.group(1)) != j:
            return None                       # non-contiguous numbering: bail loudly
        s = m.end()
        e = hits[j + 1].start() if j + 1 < len(hits) else len(tail)
        out.append(tail[s:e].strip())
    return out


def main(pats):
    n = qn = 0
    st = Counter()
    bad_examples = []
    for pat in pats:
        for path in sorted(glob.glob(pat)):
            for line in gzip.open(path, "rt"):
                d = json.loads(line)
                cands = d.get("cands") or []
                n += 1
                menu = parse_menu(d.get("prompt", ""))
                has_q = bool(d.get("qvals"))
                if has_q:
                    qn += 1
                tag = "Q" if has_q else "noQ"
                if menu is None:
                    st[tag + "/unparsable_menu"] += 1
                    continue
                if len(menu) != len(cands):
                    st[tag + "/len_mismatch"] += 1
                    if len(bad_examples) < 3:
                        bad_examples.append((len(menu), len(cands), menu[:3], cands[:3]))
                    continue
                if menu == cands:
                    st[tag + "/EXACT"] += 1
                else:
                    st[tag + "/same_len_diff_text"] += 1
                    if len(bad_examples) < 3:
                        bad_examples.append((len(menu), len(cands), menu[:3], cands[:3]))
    print("records %d (with qvals %d)" % (n, qn))
    for k, v in sorted(st.items()):
        print("  %-28s %7d  %5.1f%%" % (k, v, 100.0 * v / max(1, n)))
    for e in bad_examples:
        print("  example: menu %d cands %d\n    menu  %r\n    cands %r" % e)


if __name__ == "__main__":
    main(sys.argv[1:])
