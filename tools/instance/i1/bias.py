"""Does stopping at 904,723 of 1,198,000 skew WHICH DECKS the model saw?

Replays the exact sampling the running job used -- read_rows(seed=1234, cap=1.2M) then
Random(0).shuffle -- and compares the deck mix of the consumed prefix against the tail.
The pilot deck comes from the state's DECK[...] segment, canonicalised to a frozenset of
(card id, count) so token ORDER cannot split one decklist into two signatures.
"""
import gzip, json, random, collections, sys
sys.path.insert(0,"/root/ptcg/repo"); sys.path.insert(0,"/root/ptcg/repo/cg-lib")
import library
PATH="/root/data/rerank/curengine_0724_none.rerank.jsonl.gz"
CAP=1_200_000; SEEN=904_723; EVAL=2000

def sig(state):
    if not state.startswith("DECK["): return None
    body=state[5:state.index("]")]
    out=[]
    for t in body.split(","):
        c,_,n=t.partition("x"); out.append((int(c[1:]), int(n or 1)))
    return frozenset(out)

rng=random.Random(1234); rows=[]
with gzip.open(PATH,"rt") as f:
    for n,line in enumerate(f):
        if len(rows)<CAP:
            rows.append(sig(json.loads(line)["state"]))
        else:
            j=rng.randrange(n+1)
            if j<CAP: rows[j]=sig(json.loads(line)["state"])
print(f"pool {len(rows)} rows, {len(set(rows))} distinct decklists")
random.Random(0).shuffle(rows)
train=rows[EVAL:]
pre=collections.Counter(train[:SEEN]); post=collections.Counter(train[SEEN:])
name={}
for nm in library.list_decks():
    try: name[frozenset(collections.Counter(library.read_deck(nm)).items())]=nm
    except Exception: pass
a_t=sum(pre.values()); b_t=sum(post.values())
out=[]
for s in set(pre)|set(post):
    a=pre.get(s,0)/a_t*100; b=post.get(s,0)/b_t*100
    out.append((abs(a-b), name.get(s,"?"), a, b, pre.get(s,0)))
out.sort(reverse=True)
print(f"consumed {a_t} ({100*a_t/(a_t+b_t):.1f}%)   remaining {b_t}")
print(f"{'deck':26s} {'seen%':>7s} {'unseen%':>8s} {'diff pp':>8s} {'seen recs':>10s}")
for d,nm,a,b,n in out[:5]+out[-3:]:
    print(f"{nm:26s} {a:7.3f} {b:8.3f} {a-b:+8.3f} {n:10d}")
print(f"\ndecks {len(out)}   max |seen%-unseen%| = {out[0][0]:.3f} pp"
      f"   min records seen by a deck = {min(x[4] for x in out)}")
ev=collections.Counter(rows[:EVAL])
print(f"EVAL SET: {len(ev)} decks, max share {100*max(ev.values())/EVAL:.2f}%, "
      f"min {100*min(ev.values())/EVAL:.2f}%, decks absent from eval = {len(out)-len(ev)}")
