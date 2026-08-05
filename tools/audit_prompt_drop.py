#!/usr/bin/env python3
"""The OTHER half of the same question: fields the observation DOES carry that the prompt drops.

tools/audit_hidden_state.py measures what the engine hides from the observation. This measures
what `lm/serialize.py` hides from the model, which is a strictly cheaper thing to fix -- no
history, no card table, just render it.

`_pk` renders id / appearThisTurn / hp / maxHp / energy TYPES / tool ids. ToJson also emits
`energyCards` (which energy CARDS are attached, not just the types they provide) and
`preEvolution` (what sits under an evolved Pokemon), and the raw observation carries `logs`.
A Special Energy's identity is not recoverable from its type: Mist Energy provides {C} and also
prevents all effects of the opponent's attacks; Legacy Energy provides {C} and denies a Prize.
Both render as the letter `C`, indistinguishable from a Basic Colorless.

    python3 tools/audit_prompt_drop.py --games 8 --opp live
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
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--seed-base", type=int, default=700000)
    ap.add_argument("--opp", default="live")
    ap.add_argument("--shard", default="")
    ap.add_argument("--min-options", type=int, default=2)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    import library
    from lm.agent import make_lm_agent
    from lm import vocab
    from tools.mirror_env import DEFAULT_SO, MirrorEngine

    eng = MirrorEngine(DEFAULT_SO)
    tuning = json.load(open(os.path.join(ROOT, "agents", "tuning.json")))
    decks = [d.strip() for d in a.decks.split(",") if d.strip()] or sorted(library.list_decks())
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        decks = decks[i::n]
    if a.opp == "live":
        from tools.rl_config import LIVE_META
        opps = [d for d, _ in sorted(LIVE_META.items(), key=lambda kv: -kv[1])[:10]
                if d in set(library.list_decks())]
    elif a.opp == "mirror":
        opps = None
    else:
        opps = [d.strip() for d in a.opp.split(",") if d.strip()]

    def load(n):
        return [int(x) for x in open(library.deck_path(n)) if x.strip()]

    # Special Energy = a card that is not a Basic Energy but is attached as one. cardType 6 is
    # SP-NRG in lm.serialize._TRAINER_KIND.
    def special(cid):
        c = vocab._CARDS.get(cid)
        return c is not None and c.cardType == 6

    tot = collections.Counter()
    by_card = collections.Counter()
    per_deck = {}
    for di, deck in enumerate(decks):
        ids = load(deck)
        agent = make_lm_agent(ids, tuning.get(deck, {}), model=None)
        st = collections.Counter()
        for g in range(a.games):
            oname = deck if opps is None else opps[g % len(opps)]
            oids = list(ids) if opps is None else load(oname)
            oagent = make_lm_agent(oids, tuning.get(oname, {}), model=None)
            obs = eng.start(ids, oids, a.seed_base + di * 1000 + g,
                            mirror=1 if opps is None else 0)
            if obs is None:
                continue
            try:
                for _ in range(4000):
                    cur = obs.get("current") or {}
                    if cur.get("result", -1) != -1 or not obs.get("select"):
                        break
                    yi = cur.get("yourIndex", 0)
                    opts = (obs.get("select") or {}).get("option") or []
                    if yi == 0 and len(opts) >= a.min_options:
                        st["decisions"] += 1
                        sp = pre = 0
                        for pi, pl in enumerate(cur.get("players") or []):
                            for z in ("active", "bench"):
                                for m in (pl.get(z) or []):
                                    if not m:
                                        continue
                                    for e in (m.get("energyCards") or []):
                                        if special(e.get("id")):
                                            sp += 1
                                            by_card[e["id"]] += 1
                                    if m.get("preEvolution"):
                                        pre += 1
                        if sp:
                            st["special_energy"] += 1
                        if pre:
                            st["pre_evolution"] += 1
                        if obs.get("logs"):
                            st["logs_nonempty"] += 1
                    obs = eng.select((agent if yi == 0 else oagent)(obs))
            except Exception:
                pass
            finally:
                eng.finish()
        per_deck[deck] = dict(st)
        tot.update(st)
        n = st["decisions"] or 1
        print("  %-24s %6d dec   special-energy on board %5.1f%%   pre-evo %5.1f%%"
              % (deck, st["decisions"], 100 * st["special_energy"] / n,
                 100 * st["pre_evolution"] / n), flush=True)

    n = tot["decisions"] or 1
    print("\n%d decisions\n" % tot["decisions"])
    for k in ("special_energy", "pre_evolution", "logs_nonempty"):
        print("%-28s%10d%7.1f%%" % (k, tot[k], 100 * tot[k] / n))
    print("\nspecial energies seen (attached, rendered as a bare type letter):")
    for cid, c in by_card.most_common(20):
        print("  c%-6d %-34s %8d" % (cid, vocab.card_name(cid)[:34], c))
    if a.out:
        json.dump({"per_deck": per_deck, "total": dict(tot),
                   "by_card": {str(k): v for k, v in by_card.items()}}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
