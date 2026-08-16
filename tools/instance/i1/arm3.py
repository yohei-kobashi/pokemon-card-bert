"""3-arm deadlock comparison at n=40/cell: is the dusknoir deadlock LM-specific or the DECK's?

The 8-game version put engine_v2 WORST (35% vs alakazam) and that is too important to leave at
n=8 -- it decides whether the fix belongs in tools/dusk_plan.py (the LM's rules) or in
agents/engine_v2.py (the heuristic every fallback tier also uses).
"""
import sys, json
sys.path.insert(0,"tools")
import mirror_match as mm
from tools.mirror_env import DEFAULT_SO, MirrorEngine, play
eng = MirrorEngine(DEFAULT_SO)
tuning = json.load(open("agents/tuning.json"))
DECK="dragapult_dusknoir"; my=mm.load_deck(DECK)
_ES={"current":None,"logs":[],"remainingOverageTime":600.0,"search_begin_input":None,"select":None,"step":1}
W="planfilter:lethal_now,spread_aim,clops_hold,energy_line,energy_focus"
N=40
def watched(ag, st, seen):
    def f(obs):
        try:
            cur=obs.get("current") or {}; pl=cur.get("players") or []; yi=cur.get("yourIndex")
            if pl and yi is not None and yi<len(pl):
                k=(st["g"],cur.get("turn"))
                if k not in seen:
                    seen.add(k); p=pl[yi]; a=p.get("active")
                    a=a[0] if isinstance(a,list) and a else a
                    if isinstance(a,dict):
                        st["turns"]+=1
                        if not (a.get("energies") or []):
                            st["dry"]+=1
                            if any(len(b.get("energies") or [])>=2 for b in (p.get("bench") or [])): st["dead"]+=1
        except Exception: pass
        return ag(obs)
    return f
print("%-7s %-18s %6s %18s %14s" % ("arm","opp","win","active_dry","DEADLOCK"), flush=True)
for label,spec,fmt in (("engine","engine","prompt"),("bare","hf:/root/out/mrl2_r5b","dusk"),("def",W+":hf:/root/out/mrl2_r5b","dusk")):
    mm._FMT=fmt
    ag,_=mm.make_agent(spec,DECK,my,tuning.get(DECK,{}))
    for opp in ("alakazam_nz","marnie_grimmsnarl"):
        oid=mm.load_deck(opp); oa,_=mm.make_agent("engine",opp,oid,tuning.get(opp,{}))
        st={"turns":0,"dry":0,"dead":0,"g":0}; seen=set(); w=0
        wa=watched(ag,st,seen)
        for g in range(N):
            st["g"]=g; wa(_ES)
            mine=g%2; s=1+g//2
            r=play(eng,wa,oa,my,oid,s,mirror=1) if mine==0 else play(eng,oa,wa,oid,my,s,mirror=1)
            w+= (r==mine)
        t=max(1,st["turns"])
        print("%-7s %-18s %2d/%-3d %8d/%-4d(%2.0f%%) %6d (%2.0f%%)"
              % (label,opp,w,N,st["dry"],st["turns"],100*st["dry"]/t,st["dead"],100*st["dead"]/t), flush=True)
