import os, sys, statistics, re
os.environ["CUDA_VISIBLE_DEVICES"]=""
ROOT=os.path.expanduser("~/ptcg/repo")
for p in (ROOT, ROOT+"/cg-lib", ROOT+"/tools"): sys.path.insert(0,p)
from transformers import AutoTokenizer
import library, arena
from lm.serialize import serialize_stateless, glossary_ids, render_card_rules, render_state, render_options
from agents._engine import _CARDS
from battle_log import load_agent
tok=AutoTokenizer.from_pretrained(ROOT+"/out/rl/sft_merged")
def nt(s): return len(tok(s,add_special_tokens=False)["input_ids"]) if s else 0
BASIC={1,2,3,4,5,6,7,8,9,10,11,12}

rows=[]
def cap(deck):
    inner=[None]
    def a(obs):
        sel=obs.get("select")
        if sel and len(sel.get("option") or [])>=2:
            ids=glossary_ids(obs, deck)
            # glossary line components
            full_gloss=0; name_toks=0; energy_line_toks=0; rule_toks=0
            for cid in ids:
                line=render_card_rules(cid)
                lt=nt(line); full_gloss+=lt
                c=_CARDS.get(cid)
                nm=(c.name if c else "")
                name_toks+=nt(" "+nm)
                if cid in BASIC: energy_line_toks+=lt
            state=render_state(obs); menu=" || "+render_options(obs)
            # discard portion of state: state has "Dm[...]" discard token multiset? measure via regex of the discard field
            rows.append(dict(gloss=full_gloss, names=name_toks, energy=energy_line_toks,
                             state=nt(state), menu=nt(menu), state_str=state))
        return inner[0](obs)
    def wrap(fn): inner[0]=fn; return a
    return wrap
pairs=[("mega_starmie","crustle_stall"),("comfey_yveltal","crustle"),("dragapult","alakazam_nz")]
for A,B in pairs:
    dl=library.read_deck(A); ol=library.read_deck(B); ag=cap(dl)(load_agent(A)); og=load_agent(B)
    for g in range(2):
        try: arena.play(ag,og,dl,ol) if g%2==0 else arena.play(og,ag,ol,dl)
        except Exception: pass
def m(k): 
    v=[r[k] for r in rows]; return statistics.mean(v) if v else 0
# discard token estimate: measure the discard substring (render_state emits discard as ",".join tokens after "D")
# rough: count tokens in the discard field via the "disc" marker if present
disc=[]
for r in rows:
    s=r["state_str"]
    mres=re.search(r'D\[[^\]]*\]', s)  # discard bracket if present
    disc.append(nt(mres.group(0)) if mres else 0)
print("decisions:",len(rows))
print(f"GLOSSARY total: {m('gloss'):.0f}  (of which card-NAMES ~{m('names'):.0f}, basic-ENERGY lines ~{m('energy'):.0f}, RULE text ~{m('gloss')-m('names')-m('energy'):.0f})")
print(f"STATE (board): {m('state'):.0f}   MENU: {m('menu'):.0f}")
print(f"discard bracket est: mean {statistics.mean(disc):.0f}  p90 {sorted(disc)[max(0,int(0.9*len(disc))-1)]:.0f}  max {max(disc) if disc else 0}")
print()
print("=== trim savings (tokens) ===")
print(f"  drop card NAMES from glossary: -{m('names'):.0f} (glossary; CACHED so mainly dec-1/reveal/training)")
print(f"  drop basic-ENERGY glossary lines: -{m('energy'):.0f} (glossary; cached)")
print(f"  ONGOING per-decision tail (state+menu) now: {m('state')+m('menu'):.0f}  <- this re-prefills EVERY decision")
