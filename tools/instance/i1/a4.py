import glob,os,sys,collections,random
ROOT="/root/ptcg/repo"
for p in (ROOT,os.path.join(ROOT,"cg-lib")): sys.path.insert(0,p)
import library
from tools.build_rerank import _read_game,_game_decks,_deck_names
files=sorted(glob.glob(os.path.join(ROOT,"data/selfplay/curengine_0724/*__vs__*.jsonl.gz")))
pat=collections.Counter(); tot=collections.Counter()
for path in random.Random(5).sample(files,8):
    seq=[]
    for header,steps in _read_game(path):
        gd=_game_decks(header,steps); dn=_deck_names(header,path)
        if len(gd)!=2 or len(dn)!=2: continue
        try: t0=collections.Counter(library.read_deck(dn[0]))
        except Exception: continue
        ok = collections.Counter(gd[0])==t0
        seq.append("." if ok else "X"); tot["ok" if ok else "swapped"]+=1
    pat[os.path.basename(path).split(".")[0][:34]]="".join(seq)
for k,v in pat.items(): print(f"  {k:36s} {v}")
print("\nover ALL games in these files:", dict(tot),
      f"-> {100*tot['swapped']/max(1,sum(tot.values())):.1f}% swapped")
