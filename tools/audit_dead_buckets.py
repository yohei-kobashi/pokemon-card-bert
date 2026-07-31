"""Fleet audit: trainers whose FUNCTION is search/draw/pivot but whose NAME misses the
keyword bucket, so no rule in decide_trainer can ever fire on them.

This is the hole that killed rockets_spidops: Team Rocket's Transceiver (a Supporter
search) was offered 246 times and played 0, because `_SEARCH_ITEMS` is a hardcoded name
list and "Team Rocket's Transceiver" is not in it. The per-deck escape hatches
(`search_items` / `draw_supporters` / `gust_cards` / `switch_cards` / `energy_accel`)
fix it one deck at a time -- this finds which decks still need one.

Static pass: classify each deck's distinct Trainers by RULE TEXT, then check whether the
name matches the bucket that a rule would gate on. A mismatch is a candidate dead card,
not a proven one (some cards are reachable through other paths), so confirm the top hits
with a play-rate probe before editing configs.

Usage:  PYTHONPATH=.:cg-lib python tools/audit_dead_buckets.py
"""
import collections
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import library                                            # noqa: E402
from agents._engine import (_CARDS, _has, _DRAW_SUPPORTERS, _GUST, _SEARCH_ITEMS,
                            _ENERGY_ACCEL, _SWITCH_CARDS, _RARE_CANDY)  # noqa: E402
from agents.engine_v2 import _RECOVERY                     # noqa: E402

# (label, text pattern, bucket it would have to be in, config key that fixes it)
RULES = [
    ("search",   re.compile(r"search your deck", re.I),      _SEARCH_ITEMS,   "search_items"),
    ("draw",     re.compile(r"\bdraw \d+ cards?|draw cards until", re.I), _DRAW_SUPPORTERS, "draw_supporters"),
    ("gust",     re.compile(r"switch (?:in )?(?:1 of )?your opponent.s benched", re.I), _GUST, "gust_cards"),
    ("pivot",    re.compile(r"switch your active", re.I),    _SWITCH_CARDS,   "switch_cards"),
    ("accel",    re.compile(r"attach .{0,40}energy .{0,30}(?:from your discard|from your deck)", re.I),
                                                             _ENERGY_ACCEL,   "energy_accel"),
    ("recover",  re.compile(r"(?:put|take) .{0,40}from your discard pile into your hand", re.I),
                                                             _RECOVERY,       None),
]


ALL_BUCKETS = (_SEARCH_ITEMS, _DRAW_SUPPORTERS, _GUST, _SWITCH_CARDS,
               _ENERGY_ACCEL, _RECOVERY, _RARE_CANDY)
E_SEARCH = re.compile(r"search your deck for .{0,30}energy", re.I)


def _text(c):
    return " ".join((getattr(s, "text", "") or "") for s in (getattr(c, "skills", None) or []))


def main():
    tun = json.load(open(os.path.join(ROOT, "agents", "tuning.json")))
    findings = collections.defaultdict(list)
    card_hits = collections.Counter()
    for name, cfg in sorted(tun.items()):
        if not isinstance(cfg, dict) or not cfg.get("archetype"):
            continue
        try:
            deck = library.read_deck(name)
        except Exception:
            continue
        if len(deck) != 60:
            continue
        counts = collections.Counter(deck)
        for cid, copies in counts.items():
            c = _CARDS.get(cid)
            if c is None or c.cardType not in (1, 2, 3, 4):     # Item/Tool/Supporter/Stadium
                continue
            txt = _text(c)
            for label, pat, bucket, key in RULES:
                if not pat.search(txt):
                    continue
                # A card matched by ANY bucket is already reachable by SOME rule --
                # checking only this label's bucket produced a flood of false positives
                # (Hilda and Petrel are in _DRAW_SUPPORTERS, Prime Catcher in _GUST,
                # Fighting Gong in _ENERGY_ACCEL, all flagged by their secondary text).
                if any(_has(c.name, b) for b in ALL_BUCKETS):
                    continue
                # Crispin-class energy-search supporters have their own path in
                # decide_trainer (_RE_E_SEARCH), so they are reachable too.
                if c.cardType == 3 and E_SEARCH.search(txt):
                    continue
                if key and cid in set(cfg.get(key) or ()):
                    continue                                    # already opted in
                findings[name].append((label, cid, c.name, copies, key))
                card_hits[(cid, c.name, label, key)] += 1
    print(f"decks with at least one unreachable search/draw/pivot trainer: "
          f"{len(findings)}\n")
    order = sorted(findings.items(), key=lambda kv: -sum(x[3] for x in kv[1]))
    for name, hits in order:
        tot = sum(h[3] for h in hits)
        print(f"  {name:22} {tot:>2} copies  " +
              ", ".join(f"{h[2]}({h[0]} x{h[3]})" for h in sorted(hits, key=lambda h: -h[3])[:4]))
    print(f"\nmost common unreachable cards across the fleet:")
    for (cid, nm, label, key), n in card_hits.most_common(20):
        print(f"   {n:>2} decks  {nm:34} [{label}]  -> config key: {key}")


if __name__ == "__main__":
    main()
