import sys, json
sys.path.insert(0,"."); sys.path.insert(0,"cg-lib")
import library
from cg.game import battle_start, battle_select, battle_finish
from lm.agent import make_lm_agent
from lm.serialize import serialize_stateless
from lm.roles import resolve
prof=json.load(open("agents/tuning.json"))
d0=library.read_deck("rockets_honchkrow"); d1=library.read_deck("alakazam")
a0=make_lm_agent("rockets_honchkrow", prof["rockets_honchkrow"], None)
a1=make_lm_agent("alakazam", prof["alakazam"], None)
obs,_=battle_start(d0,d1)
R=resolve(prof["rockets_honchkrow"])
try:
    for i in range(4000):
        cur=obs.get("current")
        if cur is None or cur.get("result",-1)!=-1: break
        sel=obs.get("select")
        if sel is None: break
        yi=cur["yourIndex"]
        if yi==0 and cur.get("turn",0)>=4 and len(sel.get("option") or [])>=3:
            old=serialize_stateless(obs, deck_ids=d0, deck_name="rockets_honchkrow",
                                    glossary="none", deck_mode="remaining", deck_shuffle=True)
            new=serialize_stateless(obs, deck_ids=d0, deck_name="rockets_honchkrow",
                                    glossary="none", deck_mode="roles", roles=R,
                                    board_facts=True, identify="op")
            print("=== v37 ==="); print(old)
            print(); print("=== v39 ==="); print(new)
            print(); print("ID ME in v37:", "ID ME" in old, "| in v39:", "ID ME" in new)
            break
        obs=battle_select((a0 if yi==0 else a1)(obs))
finally:
    battle_finish()
