"""engine_v2 baseline for ONE deck against the 3 protocol opponents, one JSON per deck.

Why a grid and not the original 3-deck baseline_cells.py: the submission unit is a DECK, so
"the LM is 11pt below engine_v2 overall" is the wrong test for the submission decision. The
right test is per-deck, and that needs a baseline for every deck we might submit, not just the
three the reranker eval happens to cover.

Built through make_lm_agent(model=None) so the pilot is constructed exactly as eval_rerank.py
builds it (same policy object, same tuning profile) with no scorer consulted -- otherwise the
delta would mix a pilot-construction difference in with the model's contribution.
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

DECK = sys.argv[1]
GAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 60
OUT = sys.argv[3]
OPPS = ["alakazam", "crustle", "dragapult"]

tun = json.load(open(os.path.join(ROOT, "agents", "tuning.json")))
dl = library.read_deck(DECK)
me = make_lm_agent(dl, profile=tun.get(DECK, {}), model=None)   # pure engine_v2

out = {}
for o in OPPS:
    ol = library.read_deck(o)
    oa = load_agent(o)
    cw = 0
    for g in range(GAMES):
        mine = g % 2                                  # alternate who goes first
        r = arena.play(me, oa, dl, ol) if mine == 0 else arena.play(oa, me, ol, dl)
        cw += (r == mine)
    out["%s vs %s" % (DECK, o)] = 100.0 * cw / GAMES
    print("  %-22s (engine_v2) vs %-12s: %d/%d = %.1f%%" % (DECK, o, cw, GAMES,
                                                            100.0 * cw / GAMES), flush=True)
json.dump(out, open(OUT, "w"))
print("DONE %s" % DECK, flush=True)
