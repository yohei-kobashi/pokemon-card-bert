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
seen = 0
try:
    for _ in range(4000):
        cur = obs.get("current")
        if cur is None or cur.get("result",-1)!=-1 or obs.get("select") is None: break
        st = cur.get("stadium") or []
        if st and seen < 4:
            seen += 1
            print("stadium:", json.dumps(st)[:300])
            print("   in my deck?", st[0]["id"] in d_me, " in opp deck?", st[0]["id"] in d_op)
        if seen >= 4: break
        obs = battle_select((a_me if cur["yourIndex"]==0 else a_op)(obs))
finally:
    battle_finish()
if not seen: print("no stadium appeared in this game")
