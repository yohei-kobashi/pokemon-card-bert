"""Was the OPPONENT half of the ID segment also wrong in v34?

`ID ME d_x a_y OP d_z:9 a_w:9` -- the ME half came from the filename (50% wrong). The OP
half comes from lm/identify.identify(st, 1-yourIndex), a Bayesian posterior over the 62
decklists computed from the opponent's REVEALED cards, so it never touches the filename.
Verified here against the now-correct `opp` field rather than argued from the code path.
"""
import gzip, json, re, collections, sys, random
sys.path.insert(0,"/root/ptcg/repo"); sys.path.insert(0,"/root/ptcg/repo/cg-lib")
import library
P="/root/data/rerank/curengine_0724_v2.rerank.jsonl.gz"
RE_OP=re.compile(r" ID (?:ME \S+(?: a_\S+)? )?OP (.*?) \|\|")
RE_T=re.compile(r"^DECK\[[^\]]*\] T(\d+)\.")
fleet=set()
import json as J
tun=J.load(open("/root/ptcg/repo/agents/tuning.json"))
for k,v in tun.items():
    if isinstance(v,dict) and v.get("archetype"): fleet.add(k)
arch={k:tun[k]["archetype"] for k in fleet}

top1=collections.Counter(); inlist=collections.Counter(); atop=collections.Counter()
byturn=collections.defaultdict(lambda:[0,0]); nofleet=collections.Counter(); noPred=0; n=0
rng=random.Random(9)
with gzip.open(P,"rt") as f:
    for line in f:
        if rng.random()>0.02: continue
        r=json.loads(line); s=r["state"]; opp=r.get("opp")
        m=RE_OP.search(s)
        if not m: continue
        seg=m.group(1)
        if opp not in fleet: nofleet[opp]+=1; continue
        n+=1
        toks=seg.split()
        ds=[t.split(":")[0][2:] for t in toks if t.startswith("d_")]
        as_=[t.split(":")[0][2:] for t in toks if t.startswith("a_")]
        if not ds: noPred+=1
        t=RE_T.match(s); turn=int(t.group(1)) if t else -1
        hit = bool(ds) and ds[0]==opp
        top1[hit]+=1
        inlist[bool(ds) and opp in ds]+=1
        atop[bool(as_) and as_[0]==arch.get(opp)]+=1
        b=byturn[min(turn,12)]; b[0]+=hit; b[1]+=1
print(f"sampled {n} records ({sum(nofleet.values())} skipped: opp not in fleet {dict(nofleet)})")
print(f"OP deck  top-1 correct : {100*top1[True]/max(1,n):.1f}%")
print(f"OP deck  in candidates : {100*inlist[True]/max(1,n):.1f}%")
print(f"OP archetype top-1     : {100*atop[True]/max(1,n):.1f}%")
print(f"records with NO OP prediction (rendered '?'): {100*noPred/max(1,n):.1f}%")
print("\nby turn:  " + "  ".join(f"T{t}:{100*b[0]/max(1,b[1]):.0f}%({b[1]})"
      for t,b in sorted(byturn.items()) if t>0))
