import glob,os,sys,collections,random
ROOT="/root/ptcg/repo"
for p in (ROOT,os.path.join(ROOT,"cg-lib")): sys.path.insert(0,p)
import library
from tools.build_rerank import _read_game,_game_decks,_deck_names
files=sorted(glob.glob(os.path.join(ROOT,"data/selfplay/curengine_0724/*__vs__*.jsonl.gz")))
old=collections.Counter(); new=collections.Counter(); mir=0
for path in random.Random(5).sample(files,40):
    n=0
    for header,steps in _read_game(path):
        gd=_game_decks(header,steps)
        if len(gd)!=2: continue
        dn_old=_deck_names(header,path); dn_new=_deck_names(header,path,gd)
        for p_ in (0,1):
            try: truth=collections.Counter(library.read_deck(dn_old[p_]))
            except Exception: continue
            old["ok" if collections.Counter(gd[p_])==truth else "WRONG"]+=1
        for p_ in (0,1):
            try: truth=collections.Counter(library.read_deck(dn_new[p_]))
            except Exception: continue
            new["ok" if collections.Counter(gd[p_])==truth else "WRONG"]+=1
        n+=1
        if n>=6: break
print("OLD (positional):", dict(old))
print("NEW (list-match):", dict(new))
