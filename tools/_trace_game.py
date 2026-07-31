"""Per-turn trace of one replay from my perspective: board, energy, damage, prizes.
    PYTHONPATH=cg-lib python tools/_trace_game.py <episode_id> <my_team_name>
"""
import sys, os, json, glob
sys.path.insert(0, "tools"); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents._engine import _CARDS

EID = sys.argv[1]; MYNAME = sys.argv[2]
path = f"scratchpad_replays/episode-{EID}-replay.json"
rep = json.load(open(path))
names = rep["info"]["TeamNames"]; mi = names.index(MYNAME); oi = 1 - mi
print("rewards", rep.get("rewards"), "| me:", names[mi], "opp:", names[oi])


def pk(p):
    if not p: return "-"
    c = _CARDS.get(p.get("id")); nm = c.name if c else p.get("id")
    dmg = p.get("damage", 0); hp = (c.hp if c else 0)
    return f"{nm}({len(p.get('energyCards',[]))}e,{dmg}/{hp}hp)"


seen_turn = {}
for step in rep["steps"]:
    cur = step[0]["observation"].get("current")
    if not cur or not cur.get("players"): continue
    t = cur.get("turn")
    if t in seen_turn: continue
    seen_turn[t] = 1
    me = cur["players"][mi]; op = cur["players"][oi]
    myact = me.get("active"); myact = myact[0] if myact else None
    opact = op.get("active"); opact = opact[0] if opact else None
    print(f"T{t} fp={cur.get('firstPlayer')} | ME act={pk(myact)} bench={len([b for b in me.get('bench',[]) if b])} "
          f"hand={me.get('handCount')} deck={me.get('deckCount')} prizeLeft={len(me.get('prize',[]))} "
          f"| OPP act={pk(opact)} bench={len([b for b in op.get('bench',[]) if b])} prizeLeft={len(op.get('prize',[]))}")
# final result
last = [s for s in rep["steps"] if s[0]['observation'].get('current')][-1]
cur = last[0]['observation']['current']
print("FINAL result field:", cur.get("result"))
