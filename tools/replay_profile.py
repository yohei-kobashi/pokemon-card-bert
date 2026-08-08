#!/usr/bin/env python3
"""What do the leaderboard's own agents DO with a deck? Read it off their replays.

WHY THIS EXISTS. We reconstructed the top-20 slowking lists from replays and then guessed at
how they are piloted -- and guessed wrong, concluding the list "assumes human sequencing our
pilot cannot do". The teams running it are AGENTS, sitting at #1 and #2. A bot pilots that
list successfully, so the sequencing is not a human faculty; it is a policy we have not
written down. The replays contain the whole policy: every step carries the full observation
INCLUDING the option menu, plus the action that was chosen. That is the same pair our own
engine sees, so their decisions can be rendered in our own option vocabulary and counted.

    PYTHONPATH=cg-lib python3 tools/replay_profile.py --require 163,115 --limit 400
    PYTHONPATH=cg-lib python3 tools/replay_profile.py --require 163 --team Majkel1337

Output is a behaviour profile: action kinds, which attacks actually get used, which cards get
played and how often per game, and -- for a combo deck -- what goes on top of the deck. Run it
against our own pilot's log (tools/lm_mirror_log.py rows) to get the diff that says what to
implement.
"""

import argparse
import collections
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CACHE = os.path.join(ROOT, "scratchpad_replays")


def _deck_of(obs):
    """Every card id the OBSERVING player has been seen holding anywhere in this observation.

    Read the player through `yourIndex`, never through the agent's position in `steps`: the
    replay stores each agent's own view, and a view's `players` list is in board order, so
    indexing it by the agent number silently profiles the OPPONENT in half the games.
    """
    cur = obs.get("current") or {}
    pl = cur.get("players") or []
    yi = cur.get("yourIndex", 0)
    if yi >= len(pl) or not pl[yi]:
        return set()
    ps = pl[yi]
    out = set()
    for zone in ("hand", "discard", "prize"):
        for c in (ps.get(zone) or []):
            if isinstance(c, dict) and c.get("id"):
                out.add(c["id"])
    for slot in list(ps.get("active") or []) + list(ps.get("bench") or []):
        if not isinstance(slot, dict):
            continue
        if slot.get("id"):
            out.add(slot["id"])
        for c in (slot.get("evolves") or []):
            if isinstance(c, dict) and c.get("id"):
                out.add(c["id"])
    return out


_TO_DECK_CTX = {9, 10}       # SelectContext.TO_DECK / TO_DECK_BOTTOM (verified)


def _resolve(o, obs):
    """Card id behind an option, INCLUDING the by-reference forms.

    A play/discard option names its card by ZONE + INDEX with cardId=None (the card lives at
    players[pi].hand[index] or sel.deck[index]); reading o["cardId"] alone made every trainer
    invisible, which is why the first profile printed an empty CARDS PLAYED while 800 plays
    per game were being counted.
    """
    if not isinstance(o, dict):
        return None
    if o.get("cardId"):
        return o["cardId"]
    cur = obs.get("current") or {}
    pl = cur.get("players") or []
    pi = o.get("playerIndex")
    pi = cur.get("yourIndex", 0) if pi is None else pi
    idx = o.get("index")
    if idx is None or pi >= len(pl) or not pl[pi]:
        return None
    area = o.get("area")
    zone = {1: "hand", 4: "deck", 5: "discard", 6: "prize"}.get(area)
    if zone == "deck":
        deck = (obs.get("select") or {}).get("deck") or []
        c = deck[idx] if idx < len(deck) else None
    else:
        seq = (pl[pi].get(zone) or []) if zone else []
        c = seq[idx] if idx < len(seq) else None
    return c.get("id") if isinstance(c, dict) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--require", default="163",
                    help="comma list of card ids the acting player must have been seen with; "
                         "this is how a replay is attributed to an archetype without trusting "
                         "the team name, which changes between scrapes")
    ap.add_argument("--team", default="", help="also require this team name")
    ap.add_argument("--limit", type=int, default=300, help="max replays to scan")
    ap.add_argument("--cache", default=CACHE)
    a = ap.parse_args()

    from lm.actions import encode_option
    from agents._engine import _CARDS, _ATTACKS
    need = {int(x) for x in a.require.split(",") if x.strip()}

    kinds = collections.Counter()
    attacks = collections.Counter()
    plays = collections.Counter()
    abilities = collections.Counter()
    to_deck = collections.Counter()
    cards = collections.Counter()
    actives = collections.Counter()
    games = decisions = 0
    # NEWEST FIRST. The cache accumulates across scrapes and name order is arbitrary, so a
    # capped scan in name order reads whichever campaign happened to sort first -- the first
    # attempt matched 0 slowking games while 201 distinct card ids were being extracted fine.
    files = sorted(glob.glob(os.path.join(a.cache, "episode-*-replay.json")),
                   key=os.path.getmtime, reverse=True)[:a.limit]
    for fn in files:
        try:
            d = json.load(open(fn))
        except Exception:                                     # noqa: BLE001
            continue
        names = ((d.get("info") or {}).get("TeamNames")) or []
        steps = d.get("steps") or []
        # Which agent index is the deck we care about? Union what each side was seen holding
        # across the whole replay, then require the marker cards.
        seen = [set(), set()]
        for pair in steps:
            for mi, ag in enumerate(pair[:2]):
                seen[mi] |= _deck_of(ag.get("observation") or {})
        who = [mi for mi in (0, 1) if need <= seen[mi]]
        if a.team:
            who = [mi for mi in who if mi < len(names) and names[mi] == a.team]
        if not who:
            continue
        mi = who[0]
        games += 1
        for pair in steps:
            if mi >= len(pair):
                continue
            ag = pair[mi]
            act = ag.get("action")
            obs = ag.get("observation") or {}
            sel = obs.get("select") or {}
            opts = sel.get("option") or []
            if not act or not opts or not isinstance(act, list):
                continue
            cur = obs.get("current") or {}
            pl = (cur.get("players") or [])
            yi = cur.get("yourIndex", 0)
            if yi < len(pl) and pl[yi]:
                act_slot = ((pl[yi].get("active") or [None]) or [None])[0]
                if isinstance(act_slot, dict) and act_slot.get("id"):
                    actives[act_slot["id"]] += 1
            for i in act:
                if not isinstance(i, int) or not (0 <= i < len(opts)):
                    continue
                o = opts[i]
                decisions += 1
                try:
                    txt = encode_option(o, obs)
                except Exception:                             # noqa: BLE001
                    txt = ""
                k = (txt.split(":")[0] if txt else "?")
                kinds[k] += 1
                aid = o.get("attackId") if isinstance(o, dict) else None
                if aid:
                    attacks[aid] += 1
                cid = _resolve(o, obs)
                if cid:
                    if k == "play":
                        plays[cid] += 1
                    elif k in ("ability", "abl"):
                        abilities[cid] += 1
                    elif k in ("card", "facedown"):
                        cards[cid] += 1
                # "put a card on top of your deck": the DESTINATION is the deck, so the option
                # names a HAND card and the select's context is the deck placement.
                if sel.get("context") in _TO_DECK_CTX and cid:
                    to_deck[cid] += 1

    nm = lambda i: (_CARDS[i].name if i in _CARDS else str(i))                # noqa: E731
    an = lambda i: (_ATTACKS[i].name if i in _ATTACKS else str(i))            # noqa: E731
    print("scanned %d replays, matched %d games, %d decisions" % (len(files), games, decisions))
    if not games:
        return
    print("\naction kinds (per game):")
    for k, v in kinds.most_common(12):
        print("   %-12s %6d  %6.2f/game" % (k, v, v / games))
    print("\nATTACKS USED (per game):")
    for i, v in attacks.most_common(12):
        print("   %-24s %6d  %6.2f/game" % (an(i), v, v / games))
    print("\nActive Pokemon at decision time (share of decisions):")
    tot = sum(actives.values()) or 1
    for i, v in actives.most_common(8):
        print("   %-24s %6d  %5.1f%%" % (nm(i), v, 100 * v / tot))
    print("\nCARDS PLAYED (per game):")
    for i, v in plays.most_common(14):
        print("   %-28s %6d  %6.2f/game" % (nm(i), v, v / games))
    if abilities:
        print("\nABILITIES USED (per game):")
        for i, v in abilities.most_common(10):
            print("   %-28s %6d  %6.2f/game" % (nm(i), v, v / games))
    if cards:
        print("\nCARD-NAMED SELECTS (search / discard / choose, per game):")
        for i, v in cards.most_common(14):
            print("   %-28s %6d  %6.2f/game" % (nm(i), v, v / games))
    if to_deck:
        print("\nPUT ON TOP OF DECK (per game):")
        for i, v in to_deck.most_common(10):
            print("   %-28s %6d  %6.2f/game" % (nm(i), v, v / games))


if __name__ == "__main__":
    main()
