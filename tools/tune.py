"""Per-deck diagnosis and A/B testing for the improvement loop.

    python tools/tune.py diagnose <deck>
        Current vs-random win rate, which Pokemon the agent actually attacks
        with, and auto-suggested main_attackers hints (0-listed-damage scaling
        ex the engine tends to ignore -- the recurring failure mode).

    python tools/tune.py test <deck> [--style S] [--main 766,123] [--games N]
        Win rate of a CANDIDATE tuning vs the current one (no files changed) --
        A/B a change before committing it to agents/tuning.json.

Loop: evaluate -> tune.py diagnose (find weak decks & causes) -> edit
agents/tuning.json -> tools/generate_agents.py -> evaluate again.
"""
import argparse
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.dirname(__file__)):
    if p not in sys.path:
        sys.path.insert(0, p)

import arena  # noqa: E402
import library  # noqa: E402
from battle_log import load_agent  # noqa: E402
from cg.api import all_card_data, all_attack, CardType  # noqa: E402
from agents._engine import act  # noqa: E402

_C = {c.cardId: c for c in all_card_data()}
_A = {a.attackId: a for a in all_attack()}


def _name(cid):
    c = _C.get(cid)
    return c.name if c else str(cid)


def _best_listed_dmg(cid):
    c = _C.get(cid)
    ds = [_A[a].damage for a in c.attacks if _A.get(a) and _A[a].damage] if c else []
    return max(ds) if ds else 0


def scaling_candidates(deck_ids):
    """Multi-copy ex whose best LISTED damage is 0 (scaling attack the engine
    ignores) -> main_attackers hint candidates. The recurring fix."""
    cnt = Counter(deck_ids)
    out = []
    for cid in cnt:
        c = _C.get(cid)
        if not c or c.cardType != CardType.POKEMON:
            continue
        has_scaling = any(_A.get(a) and (_A[a].damage in (0, None)) and _A[a].energies
                          for a in c.attacks)
        if (c.ex or c.megaEx) and _best_listed_dmg(cid) == 0 and has_scaling:
            out.append((cid, c.name, cnt[cid]))
    return out


def attacker_usage(agent, deck, games=6):
    """Counter of which Pokemon (by name) the agent attacks with."""
    from cg.game import battle_start, battle_select, battle_finish
    from cg.api import OptionType
    import random
    usage = Counter()
    for seed in range(games):
        random.seed(seed)
        obs, sd = battle_start(deck, deck)
        if obs is None:
            continue
        try:
            for _ in range(4000):
                cur = obs.get("current")
                if cur is None or cur.get("result", -1) != -1:
                    break
                yi = cur["yourIndex"]; sel = obs["select"]
                ch = agent(obs) if yi == 0 else arena.random_policy(obs)
                if yi == 0 and ch and sel["option"][ch[0]]["type"] == OptionType.ATTACK:
                    me = cur["players"][0]
                    ac = me["active"][0] if me["active"] else None
                    usage[_name(ac["id"]) if ac else "?"] += 1
                obs = battle_select(ch)
        finally:
            battle_finish()
    return usage


def _mk_agent(deck, style, main):
    hints = {"main_attackers": set(main)} if main else None
    return lambda o: act(o, deck, style, hints)


def diagnose(name, games=60):
    deck = library.read_deck(name)
    agent = load_agent(name)
    w, p = arena.winrate_vs_random(agent, deck, games=games)
    print(f"== {name} ==")
    print(f"  win rate vs random: {100 * w / p:.0f}%  ({w}/{p}, {games} games)")
    usage = attacker_usage(agent, deck)
    print(f"  attacks with: {dict(usage.most_common())}")
    cand = scaling_candidates(deck)
    if cand:
        print("  scaling-ex (0 listed dmg -> engine ignores unless hinted):")
        for cid, nm, n in cand:
            used = usage.get(nm, 0)
            flag = "OK (used)" if used else "NOT USED -> hint main_attackers"
            print(f"     id{cid} {nm} x{n}  -> {flag}")
        ids = ",".join(str(c[0]) for c in cand)
        print(f"  try: python tools/tune.py test {name} --main {ids}")
    else:
        print("  no 0-damage scaling ex; if weak, try --style spread/aggro or "
              "--main <your main attacker id>")


def test(name, style=None, main=None, games=120):
    deck = library.read_deck(name)
    # baseline = the deck's current baked agent
    base = load_agent(name)
    wb, pb = arena.winrate_vs_random(base, deck, games=games)
    cand = _mk_agent(deck, style or _cur_style(name), main)
    wc, pc = arena.winrate_vs_random(cand, deck, games=games)
    b = 100 * wb / pb if pb else 0
    c = 100 * wc / pc if pc else 0
    print(f"== {name} A/B ({games} games) ==")
    print(f"  current           : {b:.0f}%")
    print(f"  candidate style={style or _cur_style(name)} main={sorted(main) if main else None}: "
          f"{c:.0f}%   ({'+' if c >= b else ''}{c - b:.0f})")
    if c > b:
        print("  -> better. Commit by editing agents/tuning.json then "
              "`python tools/generate_agents.py`")


def _cur_style(name):
    import json
    t = json.load(open(os.path.join(ROOT, "agents", "tuning.json")))
    return t.get(name, {}).get("style", "aggro")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("diagnose"); d.add_argument("deck"); d.add_argument("--games", type=int, default=60)
    t = sub.add_parser("test"); t.add_argument("deck")
    t.add_argument("--style", default=None); t.add_argument("--main", default="")
    t.add_argument("--games", type=int, default=120)
    a = ap.parse_args()
    if a.cmd == "diagnose":
        diagnose(a.deck, games=a.games)
    else:
        main_ids = [int(x) for x in a.main.split(",") if x.strip()]
        test(a.deck, style=a.style, main=main_ids or None, games=a.games)


if __name__ == "__main__":
    main()
