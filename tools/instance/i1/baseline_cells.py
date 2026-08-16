"""The engine_v2-vs-engine_v2 baseline for the SAME 9 cells the reranker is scored on.

Without it, "the reranker wins 53.3%" is uninterpretable: these are three chosen decks
against three chosen opponents, and the matchups are not balanced by construction (a deck
that beats dragapult 83% would do so with no model at all). What we actually want to know is
the DELTA the model contributes, cell by cell.

Built through make_lm_agent(model=None) rather than load_agent so the pilot is constructed
exactly as it is in eval_rerank.py -- same policy object, same tuning profile, the only
difference being that no scorer is consulted.
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

GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 60
DECKS = ["mega_lucario", "alakazam_nz_fez", "crustle_stall"]
OPPS = ["alakazam", "crustle", "dragapult"]

tun = json.load(open(os.path.join(ROOT, "agents", "tuning.json")))
w = n = 0
out = {}
for d in DECKS:
    dl = library.read_deck(d)
    me = make_lm_agent(dl, profile=tun.get(d, {}), model=None)     # pure engine_v2
    for o in OPPS:
        ol = library.read_deck(o)
        oa = load_agent(o)
        cw = 0
        for g in range(GAMES):
            mine = g % 2
            r = (arena.play(me, oa, dl, ol) if mine == 0 else arena.play(oa, me, ol, dl))
            cw += (r == mine)
        out[f"{d} vs {o}"] = 100.0 * cw / GAMES
        w += cw
        n += GAMES
        print(f"  {d:20} (engine_v2) vs {o:20}: {cw}/{GAMES} = {100.0 * cw / GAMES:.1f}%",
              flush=True)
print(f"OVERALL engine_v2 baseline {w}/{n} = {100.0 * w / n:.1f}%")
json.dump(out, open("/root/out/wr_baseline.json", "w"))
