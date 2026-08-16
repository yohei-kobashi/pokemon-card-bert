"""Fleet audit, DEFECT 1: a Basic tiered BELOW the evolution it feeds.

`_card_need` ranks search targets by `card_roles` -> `_TIER_VALUE`
(win 900 / engine 700 / line 600 / fuel 400 / tech 200 / filler 50), and a card with no explicit
role falls back to `_card_usefulness` (Pokemon ~50-60), i.e. below EVERY tier. Searches fetch from
the top. So if a deck tiers its Stage 1/2 above the Basic that evolves into it, every
"search your deck for a Pokemon" fetches an UNPLAYABLE evolution instead of a benchable body.

Measured on rockets_honchkrow: Honchkrow/Porygon2 = win(900), Murkrow/Porygon = line(600),
Articuno = tech(200). Promoting the three Basics cut bench-out 39.4% -> 22.3%
([[honchkrow-profile-was-net-negative]]).

A deck is only EXPOSED if it can actually fetch a Pokemon by tier, so each hit is also marked
with whether the deck runs a Pokemon-search card (rule text "search your deck for ... Pokemon").

Run:  PYTHONPATH=.:cg-lib python audit_tier_inversion.py
"""
import collections
import json
import os
import re
import sys

ROOT = os.environ.get("PROBE_ROOT", "/root/ptcg/repo")
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(ROOT)

import library                                                    # noqa: E402
from agents.engine_v2 import _CARDS, _TIER_VALUE                  # noqa: E402

UNTIERED = 55          # _card_usefulness for a Pokemon, i.e. below every tier
RE_PK_SEARCH = re.compile(r"search your deck for[^.]{0,80}pok", re.I | re.S)


def _norm(s):
    return (s or "").replace("’", "'").strip().lower()


def _text(c):
    out = []
    for s in (c.skills or []):
        t = getattr(s, "text", None) or getattr(s, "description", None) or ""
        out.append(str(t))
    return " ".join(out)


def tier(cid, roles):
    r = roles.get(str(cid))
    return _TIER_VALUE.get(r, UNTIERED) if r else UNTIERED, (r or "-")


def main():
    prof = json.load(open("agents/tuning.json"))
    rows = []
    for deck in sorted(prof):
        try:
            ids = library.read_deck(deck)
        except Exception:
            continue
        cnt = collections.Counter(ids)
        roles = prof[deck].get("card_roles") or {}
        cards = {i: _CARDS[i] for i in cnt if _CARDS.get(i) is not None}
        by_name = {}
        for i, c in cards.items():
            by_name.setdefault(_norm(c.name), i)
        # can this deck fetch a Pokemon by tier at all?
        fetchers = [c.name for c in cards.values() if RE_PK_SEARCH.search(_text(c))]
        bad = []
        for i, c in cards.items():
            if not (c.stage1 or c.stage2):
                continue
            src = by_name.get(_norm(c.evolvesFrom))
            if src is None:
                continue                     # evolves from something not in this deck
            hi, hr = tier(i, roles)
            lo, lr = tier(src, roles)
            if lo < hi:
                bad.append((hi - lo, cards[src].name, lr, lo, c.name, hr, hi,
                            cnt[src], cnt[i]))
        if bad:
            rows.append((max(b[0] for b in bad), deck, sorted(bad, reverse=True),
                         len(fetchers), fetchers[:3]))
    rows.sort(reverse=True)
    print("DEFECT 1 -- lower stage tiered BELOW its own evolution")
    print("%d of %d decks affected\n" % (len(rows), len(prof)))
    print("  %-24s %5s %6s  %s" % ("deck", "gap", "fetch", "worst inversion"))
    for gap, deck, bad, nf, names in rows:
        g, lname, lr, lo, hname, hr, hi, ln, hn = bad[0]
        print("  %-24s %5d %6s  %dx %s [%s %d]  <  %dx %s [%s %d]%s"
              % (deck, gap, ("YES" if nf else "no"), ln, lname[:22], lr, lo,
                 hn, hname[:22], hr, hi,
                 ("   (+%d more)" % (len(bad) - 1)) if len(bad) > 1 else ""))
    exposed = [r for r in rows if r[3]]
    print("\n  EXPOSED (inversion AND a Pokemon-search card): %d decks" % len(exposed))
    for gap, deck, bad, nf, names in exposed:
        print("    %-24s fetchers: %s" % (deck, ", ".join(n[:26] for n in names)))


if __name__ == "__main__":
    main()
