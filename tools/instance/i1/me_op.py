"""In the v34 data, how often did the buggy `ME d_x` coincide with the OP prediction?

If the swapped half made ME agree with OP, the model was not shown "noise" -- it was shown a
CORRECT opponent label half the time, which is a feature worth learning and one that inverts
its meaning at deploy (where ME is always OUR deck and can never match OP).
"""
import gzip, json, re, collections, random, sys
P="/root/data/rerank/curengine_0724_none.rerank.jsonl.gz"      # the buggy build
RE=re.compile(r" ID ME (d_\S+)(?: (a_\S+))? OP (.*?) \|\|")
n=agree=noPred=0; rng=random.Random(9)
byturn=collections.defaultdict(lambda:[0,0])
RE_T=re.compile(r"^DECK\[[^\]]*\] T(\d+)\.")
with gzip.open(P,"rt") as f:
    for line in f:
        if rng.random()>0.02: continue
        s=json.loads(line)["state"]
        m=RE.search(s)
        if not m: continue
        me=m.group(1); ds=[t.split(":")[0] for t in m.group(3).split() if t.startswith("d_")]
        n+=1
        if not ds: noPred+=1; continue
        a = (me==ds[0])
        agree+=a
        t=RE_T.match(s); turn=int(t.group(1)) if t else -1
        b=byturn[min(turn,12)]; b[0]+=a; b[1]+=1
print(f"sampled {n} records ({100*noPred/max(1,n):.1f}% had no OP prediction)")
print(f"ME token == OP top-1 prediction : {100*agree/max(1,n-noPred):.1f}%")
print("by turn: " + "  ".join(f"T{t}:{100*b[0]/max(1,b[1]):.0f}%" for t,b in sorted(byturn.items()) if t>0))
