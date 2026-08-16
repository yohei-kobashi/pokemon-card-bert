import glob, os, sys, random, collections, json
ROOT="/root/ptcg/repo"
for p in (ROOT, os.path.join(ROOT,"cg-lib")): sys.path.insert(0,p)
from lm import vocab
from lm.serialize import _pk
from tools.build_rerank import _read_game

files=sorted(glob.glob(os.path.join(ROOT,"data/selfplay/curengine_0724/*__vs__*.jsonl.gz")))
sel=random.Random(11).sample(files,12)
sp=collections.Counter(); mism=collections.Counter(); ex=[]
for path in sel:
    g=0
    for header,steps in _read_game(path):
        for s in steps:
            cur=(s.get("obs") or {}).get("current") or {}
            for pl in (cur.get("players") or []):
                for z in ("active","bench"):
                    for pkm in (pl.get(z) or []):
                        if not isinstance(pkm,dict): continue
                        ecs=pkm.get("energyCards") or []; es=pkm.get("energies") or []
                        for ec in ecs:
                            cid=ec.get("id") if isinstance(ec,dict) else ec
                            c=vocab.card(cid)
                            if getattr(c,"cardType",None)==6: sp[(cid,getattr(c,"name",""))]+=1
                        if ecs and any(getattr(vocab.card(e.get("id") if isinstance(e,dict) else e),"cardType",None)==6 for e in ecs):
                            if len(ex)<6: ex.append((_pk(pkm), [ (e.get("id") if isinstance(e,dict) else e) for e in ecs], es))
                        mism[(len(ecs),len(es))]+=1
        g+=1
        if g>=3: break
print("special energies on board:"); 
for (cid,nm),n in sp.most_common(12): print(f"   c{cid:5d} {nm[:40]:40s} {n}")
print("\nlen(energyCards) vs len(energies):", dict(list(mism.items())[:10]))
print("\nHOW A SPECIAL-ENERGY POKEMON RENDERS TODAY:")
for r,ids,es in ex: print("   _pk ->", r, "  | actual energy CARD ids:", ids, " energies:", es)
