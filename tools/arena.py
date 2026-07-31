"""Low-level self-play arena for the cabt engine.

Drives two agent functions through a full battle via the cg C-library and
returns the winner. Used by tools/evaluate.py (round-robin) and tools/tune.py.

The cg library keeps a single global battle pointer, so one process plays one
battle at a time (evaluate.py parallelises across *processes*, not threads).
"""
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib")):
    if p not in sys.path:
        sys.path.insert(0, p)

from cg.game import battle_start, battle_select, battle_finish  # noqa: E402


def random_policy(obs):
    """A legal random move (baseline opponent)."""
    sel = obs.get("select")
    if sel is None:
        return []
    n = len(sel["option"])
    k = min(max(sel["minCount"], min(sel["maxCount"], 1)), n)
    return random.sample(range(n), k) if k > 0 else []


def play(agent0, agent1, deck0, deck1, max_steps=4000):
    """Play one battle. Returns winner index (0/1), or None for a draw/timeout.

    An agent that returns an illegal selection forfeits that game (the engine
    raises), so a buggy agent loses rather than crashing the run.
    """
    if len(deck0) != 60 or len(deck1) != 60:
        return None
    obs, sd = battle_start(deck0, deck1)
    if obs is None:
        return None
    agents = (agent0, agent1)
    try:
        for _ in range(max_steps):
            cur = obs.get("current")
            if cur is None:
                return None
            if cur.get("result", -1) != -1:
                return cur["result"]
            sel = obs.get("select")
            if sel is None:
                return None
            yi = cur["yourIndex"]
            try:
                choice = agents[yi](obs)
                obs = battle_select(choice)
            except Exception:
                return 1 - yi  # illegal move / crash -> that player forfeits
        return None
    finally:
        battle_finish()


def match(agentA, deckA, agentB, deckB, games=30):
    """Play `games` battles, alternating who goes first. Returns (winsA, winsB)."""
    wa = wb = 0
    for g in range(games):
        if g % 2 == 0:
            r = play(agentA, agentB, deckA, deckB)
            wa += (r == 0); wb += (r == 1)
        else:
            r = play(agentB, agentA, deckB, deckA)
            wb += (r == 0); wa += (r == 1)
    return wa, wb


def winrate_vs_random(agent, deck, games=40):
    """Win rate of `agent` piloting `deck` against the random baseline."""
    wins, played = 0, 0
    for g in range(games):
        r = play(agent, random_policy, deck, deck) if g % 2 == 0 \
            else play(random_policy, agent, deck, deck)
        me = 0 if g % 2 == 0 else 1
        if r is not None:
            played += 1
            wins += (r == me)
    return wins, played
