"""engine_v2 baseline for ONE (deck, opponent) cell -- the control for a live-field LM cell.

One cell per process so an arbitrary opponent list parallelises without editing the script,
and so a crash costs one matchup instead of the grid.
"""
import json
import os
import sys

ROOT = "/root/ptcg/repo"
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
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
json.dump({"%s vs %s" % (DECK, OPP): 100.0 * w / GAMES}, open(OUT, "w"))
print("  %-16s (engine_v2) vs %-20s: %d/%d = %.1f%%"
      % (DECK, OPP, w, GAMES, 100.0 * w / GAMES), flush=True)
