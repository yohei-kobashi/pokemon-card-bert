import gzip,json,glob,os,collections,random,sys
ROOT="/root/ptcg/repo"
for p in (ROOT,os.path.join(ROOT,"cg-lib")): sys.path.insert(0,p)
import library
from tools.build_rerank import _read_game,_game_decks,_deck_names
for tag in ("curengine_0724","v34_full"):
    files=sorted(glob.glob(f"{ROOT}/data/selfplay/{tag}/*__vs__*.jsonl.gz"))
    ag=collections.Counter(); fix=collections.Counter(); ngames=0
    for path in random.Random(5).sample(files,15):
        for header,steps in _read_game(path):
            gd=_game_decks(header,steps); ngames+=1
            if len(gd)!=2: continue
            agents={int(k):v for k,v in (header.get("agents") or {}).items()}
            dn=_deck_names(header,path,gd)
            for p_ in (0,1):
                try: truth=collections.Counter(library.read_deck(agents[p_]))
                except Exception: continue
                ag["ok" if collections.Counter(gd[p_])==truth else "WRONG"]+=1
                fix["ok" if dn.get(p_)==agents[p_] else "differs"]+=1
    mt=max(os.path.getmtime(f) for f in files[:50])
    import datetime
    print(f"{tag:16s} games/15files={ngames:5d}  header 'agents' vs decks: {dict(ag)}"
          f"   my fix agrees with 'agents': {dict(fix)}"
          f"   newest file {datetime.datetime.fromtimestamp(mt):%Y-%m-%d %H:%M}")
