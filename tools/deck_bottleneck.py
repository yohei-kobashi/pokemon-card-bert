"""Per-deck bottleneck diagnostic: WHY is this deck losing?

Applies the project's 3-state discriminator to the win condition, per TURN (not per
menu -- a turn holds many menus and the attack is the last one, so per-menu sampling
under-counts attacks massively), plus setup and dead-card measures:

  win body never in play      -> DECK (can't find it)
  in play but attack unpayable-> DECK (can't fuel it)
  payable but not used        -> AGENT (won't fire it)

Always run with a CONTROL BAND of known-strong decks: every one of these numbers looks
alarming in isolation and only means something next to a deck that wins.

Usage:
    PYTHONPATH=.:cg-lib python tools/deck_bottleneck.py deck1,deck2 --opp mega_lucario
"""
import argparse
import collections
import json
import os
import statistics
import sys
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)


def probe(job):
    deck, opp, games = job
    import library
    from agents.engine_v2 import make_policy
    from agents._engine import _CARDS
    from cg.game import battle_start, battle_select, battle_finish
    T = json.load(open(os.path.join(ROOT, "agents", "tuning.json")))
    cfg = T.get(deck, {})
    wins = {int(k) for k, v in (cfg.get("card_roles") or {}).items() if v == "win"}
    dl = library.read_deck(deck)
    ol = library.read_deck(opp)
    if not wins:
        wins = set(cfg.get("main_attackers") or [])

    st = collections.Counter()
    offers, plays = collections.Counter(), collections.Counter()
    turns_per_game, first_atk, energy_end = [], [], []
    for g in range(games):
        pa = make_policy(dl, cfg)
        pb = make_policy(ol, T.get(opp, {}))
        obs, _ = battle_start(dl, ol)
        seen_turn = {}          # turn -> dict of per-turn flags
        fa = None
        try:
            for _ in range(3000):
                cur = obs.get("current")
                if cur is None or cur.get("result", -1) != -1:
                    break
                sel = obs.get("select")
                if sel is None:
                    break
                yi = cur["yourIndex"]
                if yi == 0 and sel.get("context") == 0:
                    t = int(cur.get("turn") or 0)
                    f = seen_turn.setdefault(t, {"inplay": 0, "payable": 0, "used": 0,
                                                 "atk_any": 0})
                    ps = cur["players"][0]
                    board = [x for x in ((ps.get("active") or []) + (ps.get("bench") or []))
                             if x]
                    if any(b.get("id") in wins for b in board):
                        f["inplay"] = 1
                    hand = ps.get("hand") or []
                    opts = sel.get("option") or []
                    for o in opts:
                        if o.get("type") == 7 and o.get("index") is not None \
                                and o["index"] < len(hand):
                            offers[hand[o["index"]].get("id")] += 1
                    # is an attack by a WIN body on the table right now?
                    act = (ps.get("active") or [None])[0]
                    for o in opts:
                        if o.get("type") == 13:
                            f["atk_any"] = 1
                            if act and act.get("id") in wins:
                                f["payable"] = 1
                    pick = pa.act(obs)
                    for j in (pick or []):
                        if j < len(opts):
                            o = opts[j]
                            if o.get("type") == 7 and o.get("index") is not None \
                                    and o["index"] < len(hand):
                                plays[hand[o["index"]].get("id")] += 1
                            if o.get("type") == 13:
                                if fa is None:
                                    fa = t
                                if act and act.get("id") in wins:
                                    f["used"] = 1
                    obs = battle_select(pick)
                    continue
                obs = battle_select((pa, pb)[yi].act(obs))
            cur = obs.get("current") or {}
            ps = (cur.get("players") or [{}])[0]
            board = [x for x in ((ps.get("active") or []) + (ps.get("bench") or [])) if x]
            energy_end.append(sum(len(b.get("energies") or []) for b in board))
        finally:
            battle_finish()
        turns_per_game.append(len(seen_turn))
        if fa is not None:
            first_atk.append(fa)
        for f in seen_turn.values():
            st["turns"] += 1
            st["inplay"] += f["inplay"]
            st["payable"] += f["payable"]
            st["used"] += f["used"]
            st["atk_any"] += f["atk_any"]
    dead = [(cid, offers[cid]) for cid in offers
            if plays[cid] == 0 and offers[cid] >= 20]
    dead.sort(key=lambda x: -x[1])
    return {
        "deck": deck, "turns": st["turns"],
        "inplay": st["inplay"], "payable": st["payable"], "used": st["used"],
        "atk_any": st["atk_any"],
        "avg_turns": statistics.mean(turns_per_game) if turns_per_game else 0,
        "first_atk": statistics.median(first_atk) if first_atk else None,
        "energy_end": statistics.mean(energy_end) if energy_end else 0,
        "dead": [(c, n, _CARDS[c].name if c in _CARDS else "?") for c, n in dead[:4]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("decks")
    ap.add_argument("--opp", default="mega_lucario")
    ap.add_argument("--games", type=int, default=24)
    args = ap.parse_args()
    jobs = [(d, args.opp, args.games) for d in args.decks.split(",")]
    with Pool(min(len(jobs), os.cpu_count())) as p:
        res = list(p.imap(probe, jobs))
    print(f"vs {args.opp}, {args.games} games each. Rates are PER TURN.\n")
    print(f"{'deck':18}{'turns':>6}{'winbody':>9}{'payable':>9}{'FIRED':>7}"
          f"{'anyatk':>8}{'1st atk':>8}{'endE':>6}")
    for r in res:
        t = max(r["turns"], 1)
        fa = r["first_atk"]
        print(f"{r['deck']:18}{r['avg_turns']:6.1f}{100*r['inplay']/t:8.0f}%"
              f"{100*r['payable']/t:8.0f}%{100*r['used']/t:6.0f}%"
              f"{100*r['atk_any']/t:7.0f}%{(fa if fa is not None else -1):8.1f}"
              f"{r['energy_end']:6.1f}")
    print("\ncards offered >=20x and NEVER played:")
    for r in res:
        if r["dead"]:
            print(f"  {r['deck']:18} " + ", ".join(f"{nm}({n})" for _c, n, nm in r["dead"]))


if __name__ == "__main__":
    main()
