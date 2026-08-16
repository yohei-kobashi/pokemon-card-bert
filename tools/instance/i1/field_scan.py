"""Which of OUR decks actually beats the CURRENT top-2 Kaggle decks, under engine_v2?

Needed the moment [[live-attack-deadlock]] showed dragapult_dusknoir sits at ~30% vs
alakazam_nz + marnie_grimmsnarl under every pilot we have. The existing anti-meta table is from
July and the meta moved since ([[meta-refresh-aug-2026]]: marnie 35.4%, Alakazam halved), so it
cannot be reused. engine_v2 on both sides: this ranks DECKS, holding the pilot fixed.

Sequential on purpose -- tools/_vs_targets.py opens a Pool sized from os.cpu_count(), which on
these boxes reports 112 against 61.4 effective cores ([[vast-cpu-quotas]]) and would fight the
brancher. engine_v2 games are ~1ms, so the whole fleet is minutes anyway.
"""
import sys, json, time
sys.path.insert(0,"tools")
import mirror_match as mm, library
from tools.mirror_env import DEFAULT_SO, MirrorEngine, play
N=int(sys.argv[1]) if len(sys.argv)>1 else 40
TARGETS=["alakazam_nz","marnie_grimmsnarl"]
eng=MirrorEngine(DEFAULT_SO); tuning=json.load(open("agents/tuning.json"))
tg={}
for t in TARGETS:
    ids=mm.load_deck(t); tg[t]=(ids, mm.make_agent("engine",t,ids,tuning.get(t,{}))[0])
rows=[]
t0=time.time()
for name in sorted(library.list_decks()):
    try:
        my=mm.load_deck(name); ag,_=mm.make_agent("engine",name,my,tuning.get(name,{}))
    except Exception as e:
        continue
    cells={}
    for t,(oid,oa) in tg.items():
        w=0
        for g in range(N):
            mine=g%2; s=1+g//2
            try:
                r=play(eng,ag,oa,my,oid,s,mirror=1) if mine==0 else play(eng,oa,ag,oid,my,s,mirror=1)
            except Exception:
                r=None
            w+=(r==mine)
        cells[t]=w
    tot=sum(cells.values()); rows.append((tot,name,cells))
    print("%-26s %s  tot %3d/%-3d = %5.1f%%" % (name,
          "  ".join("%s %2d/%-3d"%(t[:12],cells[t],N) for t in TARGETS),
          tot, 2*N, 100.0*tot/(2*N)), flush=True)
rows.sort(reverse=True)
print("\n=== TOP 15 vs the current top-2 (engine_v2 both sides, %d games/cell, %.1f min) ===" % (N,(time.time()-t0)/60))
for tot,name,cells in rows[:15]:
    print("  %-26s %5.1f%%   (%s)" % (name, 100.0*tot/(2*N),
          " ".join("%s=%d"%(t[:10],cells[t]) for t in TARGETS)))
json.dump([{"deck":n,"cells":c,"total":t} for t,n,c in rows], open("/root/field_scan.json","w"), indent=1)
