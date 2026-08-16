import os, sys, json, statistics
os.environ["CUDA_VISIBLE_DEVICES"]=""
ROOT=os.path.expanduser("~/ptcg/repo")
for p in (ROOT, ROOT+"/cg-lib", ROOT+"/tools"): sys.path.insert(0,p)
from transformers import AutoTokenizer
import library, arena
from lm.serialize import serialize_stateless, render_card_rules, visible_card_ids, render_state, render_options
from battle_log import load_agent
tok=AutoTokenizer.from_pretrained(ROOT+"/out/rl/sft_merged")
def ntok(s): return len(tok(s,add_special_tokens=False)["input_ids"]) if s else 0

rows=[]  # (gloss_toks, state_toks, menu_toks, gloss_str)
def cap(inner):
    def a(obs):
        sel=obs.get("select")
        if sel and len(sel.get("option") or [])>=2:
            rule_ids=list(dict.fromkeys(visible_card_ids(obs)))
            rules="\n".join(render_card_rules(c) for c in rule_ids)
            gloss=("RULES "+rules+"\n") if rules else ""
            rows.append((ntok(gloss), ntok(render_state(obs)), ntok(" || "+render_options(obs)), gloss))
        return inner(obs)
    return a
pairs=[("crustle_stall","alakazam"),("dragapult","marnie_grimmsnarl"),
       ("comfey_yveltal","crustle_stall"),("alakazam","rockets_mewtwo")]
for a,b in pairs:
    dl=library.read_deck(a); ol=library.read_deck(b); ag=cap(load_agent(a)); og=load_agent(b)
    for g in range(2):
        try: arena.play(ag,og,dl,ol) if g%2==0 else arena.play(og,ag,ol,dl)
        except Exception as e: pass
gl=[r[0] for r in rows]; stt=[r[1] for r in rows]; mn=[r[2] for r in rows]; tot=[sum(r[:3]) for r in rows]
prev=None; stable=0
for r in rows:
    if r[3] and r[3]==prev: stable+=1
    prev=r[3]
def st(x): return f"mean {statistics.mean(x):.0f} p50 {statistics.median(x):.0f} p90 {sorted(x)[max(0,int(0.9*len(x))-1)]:.0f} max {max(x)}"
print("decisions:",len(rows))
print("TOTAL    :",st(tot))
print("GLOSSARY :",st(gl), f" ({100*statistics.mean(gl)/max(1,statistics.mean(tot)):.0f}%)")
print("STATE    :",st(stt),f" ({100*statistics.mean(stt)/max(1,statistics.mean(tot)):.0f}%)")
print("MENU     :",st(mn), f" ({100*statistics.mean(mn)/max(1,statistics.mean(tot)):.0f}%)")
print(f"glossary identical to prev decision: {stable}/{len(rows)} = {100*stable/max(1,len(rows)):.0f}%")
