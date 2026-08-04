#!/usr/bin/env python3
"""Check a computed live cost against the ONE oracle available at inference: the menu.

If the engine put `retreat` on the menu, the live cost is payable right now, so

    effective_retreat_cost(...) <= attached energy

must hold. That gives a fleet-wide correctness test for `lm/costs.py` with no Search API, no
extra games and no hand-checking of card text -- and it is the same test that exposed the
defect: the PRINTED cost violates it on 43% of ns_zoroark's offered retreats.

The converse is not testable this way. Retreat can be missing from the menu because the bench is
empty, because an effect forbids it, or because it is unaffordable, and the observation does not
say which -- so a cost that is too LOW is only reported, never asserted.

    python3 tools/audit_costs.py --games 12                 # every deck
    python3 tools/audit_costs.py --decks ns_zoroark,ethan_hooh --games 20
"""

import argparse
import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib")):
    if p not in sys.path:
        sys.path.insert(0, p)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decks", default="")
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--seed-base", type=int, default=100000)
    ap.add_argument("--mirror-so", default="")
    ap.add_argument("--examples", type=int, default=5)
    a = ap.parse_args()

    import library
    from lm.actions import encode_option
    from lm.agent import make_lm_agent
    from lm.costs import effective_retreat_cost
    from lm import vocab
    from tools.mirror_env import DEFAULT_SO, MirrorEngine

    eng = MirrorEngine(a.mirror_so or DEFAULT_SO)
    tuning = json.load(open(os.path.join(ROOT, "agents", "tuning.json")))
    decks = [d.strip() for d in a.decks.split(",") if d.strip()] or sorted(library.list_decks())

    tot = collections.Counter()
    rows = []
    bad_examples = []
    for di, deck in enumerate(decks):
        ids = [int(x) for x in open(library.deck_path(deck)) if x.strip()]
        agent = make_lm_agent(ids, tuning.get(deck, {}), model=None)
        st = collections.Counter()
        for g in range(a.games):
            obs = eng.start(ids, list(ids), a.seed_base + di * 1000 + g, mirror=0)
            if obs is None:
                continue
            try:
                for _ in range(4000):
                    cur = obs.get("current") or {}
                    if cur.get("result", -1) != -1 or not obs.get("select"):
                        break
                    opts = (obs.get("select") or {}).get("option") or []
                    if any(encode_option(o, obs) == "retreat" for o in opts):
                        yi = cur.get("yourIndex", 0)
                        mon = ((cur.get("players") or [])[yi].get("active") or [None])[0]
                        card = vocab._CARDS.get((mon or {}).get("id"))
                        if mon and card is not None and card.retreatCost is not None:
                            ne = len(mon.get("energies") or [])
                            st["offered"] += 1
                            if ne < card.retreatCost:
                                st["printed_violates"] += 1
                            live = effective_retreat_cost(obs, yi, mon)
                            if live is None:
                                st["live_unknown"] += 1
                            elif ne < live:
                                st["live_violates"] += 1
                                if len(bad_examples) < a.examples:
                                    bad_examples.append(
                                        (deck, mon.get("id"), getattr(card, "name", "?"),
                                         card.retreatCost, live, ne,
                                         _ids_of(cur.get("stadium")),
                                         _ids_of(mon.get("tools"))))
                    obs = eng.select(agent(obs))
            except Exception:
                pass
            finally:
                eng.finish()
        if st["offered"]:
            rows.append((deck, st["offered"], st["printed_violates"], st["live_violates"],
                         st["live_unknown"]))
            tot.update(st)

    rows.sort(key=lambda r: -(r[3] / max(1, r[1])))
    print("%-24s%9s%12s%12s%10s" % ("deck", "offered", "printed X", "live X", "unknown"))
    for deck, off, pv, lv, unk in rows:
        mark = "   <-- still wrong" if lv else ""
        print("%-24s%9d%8d %3.0f%%%8d %3.0f%%%10d%s"
              % (deck, off, pv, 100 * pv / off, lv, 100 * lv / off, unk, mark))
    o = tot["offered"] or 1
    print("\nFLEET  offered %d | PRINTED cost violates the menu %d (%.1f%%) "
          "| LIVE cost violates %d (%.1f%%) | unknown %d"
          % (tot["offered"], tot["printed_violates"], 100 * tot["printed_violates"] / o,
             tot["live_violates"], 100 * tot["live_violates"] / o, tot["live_unknown"]))
    if bad_examples:
        print("\nresidual examples (live cost still says unaffordable):")
        for deck, cid, name, printed, live, ne, stad, tools in bad_examples:
            print("  %-20s c%-6d %-24s printed %d live %d energy %d stad %s tools %s"
                  % (deck, cid, name[:24], printed, live, ne, stad, tools))


def _ids_of(seq):
    out = []
    for x in (seq or []):
        if isinstance(x, dict) and x.get("id") is not None:
            out.append(x["id"])
        elif isinstance(x, int):
            out.append(x)
    return out


if __name__ == "__main__":
    main()
