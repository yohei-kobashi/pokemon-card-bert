import os, sys, json, statistics
os.environ["CUDA_VISIBLE_DEVICES"]=""
ROOT=os.path.expanduser("~/ptcg/repo")
for p in (ROOT, ROOT+"/cg-lib", ROOT+"/tools"): sys.path.insert(0,p)
from transformers import AutoTokenizer
import library, arena
from lm.serialize import serialize_stateless, render_card_rules, visible_card_ids
from battle_log import load_agent
tok=AutoTokenizer.from_pretrained(ROOT+"/out/rl/sft_merged")

prompts=[]
def cap(inner):
    def a(obs):
        sel=obs.get("select")
        if sel and len(sel.get("option") or [])>=2:
            prompts.append(serialize_stateless(obs))
        return inner(obs)
    return a

pairs=[("crustle_stall","alakazam"),("dragapult","marnie_grimmsnarl"),
       ("comfey_yveltal","crustle_stall"),("alakazam","rockets_mewtwo")]
for a,b in pairs:
    dl=library.read_deck(a); ol=library.read_deck(b)
    ag=cap(load_agent(a)); og=load_agent(b)
    for g in range(2):
        try:
            arena.play(ag,og,dl,ol) if g%2==0 else arena.play(og,ag,ol,dl)
        except Exception as e: print("game err",e)

def headlen(pr):
    # reconstruct glossary head exactly as serialize_stateless does
    if not pr.startswith("RULES "): return 0
    # head ends right before render_state which starts with "A["; find " || " menu sep is later
    # easier: glossary = everything up to the last "\n" before the state; state has no leading RULES
    i=pr.rfind("\nA[")  # state begins "A[" on its own start after head's trailing \n
    return i+1 if i>=0 else 0

gl=[];rest=[];tot=[];prev=None;stable=0
for pr in prompts:
    hl=headlen(pr); gloss=pr[:hl]; body=pr[hl:]
    tg=len(tok(gloss,add_special_tokens=False)["input_ids"]) if gloss else 0
    tb=len(tok(body,add_special_tokens=False)["input_ids"])
    gl.append(tg);rest.append(tb);tot.append(tg+tb)
    if gloss and gloss==prev: stable+=1
    prev=gloss
def st(x): return f"mean {statistics.mean(x):.0f}  p50 {statistics.median(x):.0f}  p90 {sorted(x)[max(0,int(0.9*len(x))-1)]:.0f}  max {max(x)}"
print("decisions measured:",len(prompts))
print("TOTAL prompt tokens :",st(tot))
print("  GLOSSARY tokens   :",st(gl),f"  ({100*statistics.mean(gl)/max(1,statistics.mean(tot)):.0f}% of prompt)")
print("  BODY(state+menu)  :",st(rest))
print(f"glossary IDENTICAL to prev decision: {stable}/{len(prompts)} = {100*stable/max(1,len(prompts)):.0f}%  (free cross-decision cache-hit rate NOW)")
