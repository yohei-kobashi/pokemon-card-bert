"""How much genuinely NEW data does changing --sample-seed buy?

At --cap-matchup 90 the reservoir takes 90 records per (deck, opponent). A matchup holding
fewer than 90 records is taken WHOLE, so a new seed re-draws exactly the same rows there.
Measures the real fresh fraction rather than assuming the 90/414 average.
"""
import sys, collections
sys.path.insert(0,"/root/ptcg/repo"); sys.path.insert(0,"/root/ptcg/repo/cg-lib")
sys.path.insert(0,"/root/ptcg/repo/tools")
from train_rerank import read_rows, row_key
P="/root/data/rerank/curengine_0724_v2.rerank.jsonl.gz"
a=read_rows(P,0,seed=1234,cap_matchup=90)
b=read_rows(P,0,seed=777, cap_matchup=90)
ka={row_key(r) for r in a}; kb={row_key(r) for r in b}
inter=len(ka&kb)
print(f"seed1234 {len(ka)}  seed777 {len(kb)}  overlap {inter} = {100*inter/len(kb):.1f}%")
print(f"-> genuinely NEW records in run 2: {len(kb)-inter} ({100*(len(kb)-inter)/len(kb):.1f}%)")
cnt=collections.Counter()
for r in a: cnt[(r["deck"],r["opp"])]+=1
small=sum(1 for v in cnt.values() if v<90)
print(f"matchups {len(cnt)}; below the cap (identical in both samples): {small} "
      f"({100*small/len(cnt):.0f}%), holding {sum(v for v in cnt.values() if v<90)} rows")
