"""engine_v2's win rate for ONE (deck, opponent) cell -- the control the RL gate subtracts.

The RL goal is "reach engine_v2's level", so the gate must be a DIFFERENCE, not an absolute
win rate. Measured 2026-07-28, absolutes are useless as a target on their own: over the
Stage-C decks engine_v2 itself averages 37.4% and manages only 24.0% with dragapult and 15.8%
with dragapult_dusknoir. A gate of "reach 40%" would be unreachable on half the grid and
trivial on the other half, and neither says anything about closing the gap to the engine.

One cell per process so an arbitrary grid parallelises and a crash costs one matchup.
Built through make_lm_agent(model=None) so the pilot is constructed exactly as eval_rerank.py
builds it -- same policy object, same tuning profile, no scorer consulted -- otherwise the
delta would fold in a pilot-construction difference.

    python tools/rl_baseline_cell.py <deck> <opp> <games> <out.json>
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import arena  # noqa: E402
import library  # noqa: E402
from battle_log import load_agent  # noqa: E402
from lm.agent import make_lm_agent  # noqa: E402

DECK, OPP = sys.argv[1], sys.argv[2]
GAMES = int(sys.argv[3])
OUT = sys.argv[4]

tun = json.load(open(os.path.join(ROOT, "agents", "tuning.json")))
dl = library.read_deck(DECK)
ol = library.read_deck(OPP)
me = make_lm_agent(dl, profile=tun.get(DECK, {}), model=None)   # pure engine_v2
oa = load_agent(OPP)

w = 0
for g in range(GAMES):
    mine = g % 2                                   # alternate the first player
    r = arena.play(me, oa, dl, ol) if mine == 0 else arena.play(oa, me, ol, dl)
    w += (r == mine)
os.makedirs(os.path.dirname(os.path.abspath(OUT)) or ".", exist_ok=True)
json.dump({"deck": DECK, "opp": OPP, "win": w, "games": GAMES,
           "win_rate": 100.0 * w / GAMES}, open(OUT, "w"))
print("  %-20s (engine_v2) vs %-20s: %d/%d = %.1f%%"
      % (DECK, OPP, w, GAMES, 100.0 * w / GAMES), flush=True)
