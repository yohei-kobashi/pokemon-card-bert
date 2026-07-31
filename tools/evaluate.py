"""Round-robin cross-play evaluation of every deck's agent.

Each deck's agent pilots its own deck against every other deck's agent, N games
per pair (alternating first player). Produces a per-deck win rate ("vs field")
ranking plus the full win-rate matrix, saved to evaluations/eval_<ts>.json.

This is the "evaluate" step of the loop:
    python tools/evaluate.py                 # all decks, 20 games/pair
    python tools/evaluate.py --games 30
    python tools/evaluate.py --decks a,b,c   # subset (also useful for A/B tests)

The cg engine is single-battle-per-process, so pairs are parallelised across
worker processes.
"""
import argparse
import itertools
import json
import os
import sys
import time
from datetime import datetime
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.dirname(__file__)):
    if p not in sys.path:
        sys.path.insert(0, p)

import _sample  # noqa: E402
import arena  # noqa: E402
import library  # noqa: E402
from battle_log import load_agent  # noqa: E402

_CACHE = {}


def _load(name):
    if name not in _CACHE:
        _CACHE[name] = (load_agent(name), library.read_deck(name))
    return _CACHE[name]


def _play_pair(args):
    a, b, games = args
    agentA, deckA = _load(a)
    agentB, deckB = _load(b)
    wa, wb = arena.match(agentA, deckA, agentB, deckB, games=games)
    return a, b, wa, wb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20, help="games per pair")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--decks", type=str, default="", help="comma-separated subset")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    decks = library.list_decks()
    if args.decks:
        want = set(args.decks.split(","))
        decks = [d for d in decks if d in want]
    pairs = [(a, b, args.games) for a, b in itertools.combinations(decks, 2)]
    print(f"{len(decks)} decks, {len(pairs)} pairs x {args.games} games "
          f"= {len(pairs) * args.games} battles on {args.workers} workers")

    wins = {d: 0 for d in decks}       # total game wins vs the field
    played = {d: 0 for d in decks}
    matrix = {d: {} for d in decks}    # matrix[a][b] = a's win% vs b
    t = time.time()
    done = 0
    with Pool(args.workers) as pool:
        for a, b, wa, wb in pool.imap_unordered(_play_pair, pairs):
            wins[a] += wa; wins[b] += wb
            played[a] += wa + wb; played[b] += wa + wb
            tot = wa + wb or 1
            matrix[a][b] = round(100 * wa / tot)
            matrix[b][a] = round(100 * wb / tot)
            done += 1
            if done % 25 == 0 or done == len(pairs):
                print(f"  {done}/{len(pairs)} pairs ({time.time() - t:.0f}s)", flush=True)

    ranking = sorted(
        ((100 * wins[d] / played[d] if played[d] else 0.0, d, wins[d], played[d] - wins[d])
         for d in decks), reverse=True)

    print("\n=== vs-field ranking (win% across all opponents) ===")
    for pct, d, w, l in ranking:
        print(f"  {pct:5.1f}%  {d:22} ({w}-{l})")

    print(_sample.banner(args.games, "games/pair"), end="")

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.out or os.path.join(ROOT, "evaluations", f"eval_{ts}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    blob = _sample.stamp({
        "timestamp": ts, "games_per_pair": args.games, "decks": decks,
        "ranking": [{"deck": d, "winrate": round(p, 1), "wins": w, "losses": l}
                    for p, d, w, l in ranking],
        "matrix": matrix,
    }, args.games, "games/pair")
    json.dump(blob, open(out, "w"), indent=2)
    print(f"\nsaved -> {out}"
          + ("" if blob["trustworthy"] else "   [stamped trustworthy=false]"))


if __name__ == "__main__":
    main()
