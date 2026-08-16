"""Print real mid-game Pokemon objects: what are energies vs energyCards vs preEvolution?"""
import json, os, sys
ROOT = "/root/ptcg/repo"
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path: sys.path.insert(0, p)
import library
from cg.game import battle_start, battle_select, battle_finish
from lm.agent import make_lm_agent

d_me = library.read_deck("alakazam"); d_op = library.read_deck("dragapult")
a_me = make_lm_agent("alakazam", None, None); a_op = make_lm_agent("dragapult", None, None)
obs, _ = battle_start(d_me, d_op)
shown = 0
try:
    for _ in range(4000):
        cur = obs.get("current")
        if cur is None or cur.get("result", -1) != -1 or obs.get("select") is None: break
        for pl in cur["players"]:
            for z in ("active", "bench"):
                for x in (pl.get(z) or []):
                    if not x: continue
                    if (x.get("energies") or x.get("energyCards") or x.get("preEvolution")) and shown < 6:
                        shown += 1
                        print("--- Pokemon id=%s hp=%s" % (x.get("id"), x.get("hp")))
                        for k in ("energies","energyCards","preEvolution","tools"):
                            print("    %-14s %s" % (k, json.dumps(x.get(k))[:220]))
        if shown >= 6: break
        obs = battle_select((a_me if cur["yourIndex"] == 0 else a_op)(obs))
finally:
    battle_finish()
