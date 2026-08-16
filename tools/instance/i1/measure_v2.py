import os, sys, statistics
os.environ["CUDA_VISIBLE_DEVICES"]=""
ROOT=os.path.expanduser("~/ptcg/repo")
for p in (ROOT, ROOT+"/cg-lib", ROOT+"/tools"): sys.path.insert(0,p)
from transformers import AutoTokenizer
import library, arena
from lm.serialize import serialize_stateless, glossary_ids, render_card_rules, render_state, render_options
from battle_log import load_agent
tok=AutoTokenizer.from_pretrained(ROOT+"/out/rl/sft_merged")
def nt(s): return len(tok(s,add_special_tokens=False)["input_ids"]) if s else 0
def gloss_str(obs, deck):
    ids=glossary_ids(obs, deck); r="\n".join(render_card_rules(c) for c in ids)
    return ("RULES "+r+"\n") if r else ""

rows=[]  # (v1gloss, v2gloss, tail, v1str, v2str)
def cap(deck):
    inner=None
    def a(obs):
        sel=obs.get("select")
        if sel and len(sel.get("option") or [])>=2:
            g1=gloss_str(obs,None); g2=gloss_str(obs,deck)
            tail=render_state(obs)+" || "+render_options(obs)
            rows.append((nt(g1),nt(g2),nt(tail),g1,g2))
        return inner(obs)
    def wrap(fn):
        nonlocal inner; inner=fn; return a
    return wrap
pairs=[("crustle_stall","alakazam"),("dragapult","marnie_grimmsnarl"),("comfey_yveltal","crustle_stall")]
for A,B in pairs:
    dl=library.read_deck(A); ol=library.read_deck(B)
    ag=cap(dl)(load_agent(A)); og=load_agent(B)
    for g in range(2):
        try: arena.play(ag,og,dl,ol) if g%2==0 else arena.play(og,ag,ol,dl)
        except Exception: pass
def st(x): return f"mean {statistics.mean(x):.0f} p50 {statistics.median(x):.0f} p90 {sorted(x)[max(0,int(0.9*len(x))-1)]:.0f}"
v1=[r[0] for r in rows]; v2=[r[1] for r in rows]; tail=[r[2] for r in rows]
# stability (identical glossary to prev decision, per game-ish -> just consecutive)
def hit(idx):
    prev=None; h=0
    for r in rows:
        if r[idx] and r[idx]==prev: h+=1
        prev=r[idx]
    return 100*h/max(1,len(rows))
print("decisions:",len(rows))
print("v1 glossary tokens:",st(v1),f"  cross-decision cache-hit {hit(3):.0f}%")
print("v2 glossary tokens:",st(v2),f"  cross-decision cache-hit {hit(4):.0f}%  <-- stable = cached whole game")
print("dynamic TAIL (state+menu):",st(tail))
print(f"=> EFFECTIVE prefill/decision: v1(no cache)~{statistics.mean([r[0]+r[2] for r in rows]):.0f}tok  v2(after dec1, tail only)~{statistics.mean(tail):.0f}tok")
kaggle=119.0
print(f"=> Kaggle @119t/s: v1 ~{statistics.mean([r[0]+r[2] for r in rows])/kaggle:.1f}s/dec  v2 ~{statistics.mean(tail)/kaggle:.1f}s/dec")
