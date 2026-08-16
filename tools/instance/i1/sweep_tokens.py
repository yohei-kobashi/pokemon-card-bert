import gzip, json, os, sys
from tokenizers import Tokenizer
TOKF="/root/out/rerank_gte_mp/tokenizer.json"
DATA="/root/data/rerank/curengine_0724_mp.rerank.jsonl.gz"
tok = Tokenizer.from_file(TOKF)
V = tok.get_vocab_size()
used=set(); seen_state=set(); texts=[]; nrec=0
def flush():
    global texts
    if not texts: return
    for e in tok.encode_batch(texts):
        used.update(e.ids)
    texts=[]
with gzip.open(DATA,"rt") as fh:
    for line in fh:
        r=json.loads(line); nrec+=1
        s=r["state"]
        h=hash(s)
        if h not in seen_state:
            seen_state.add(h); texts.append(s)
        for c in r["candidates"]:
            texts.append(c)
        if len(texts)>=4096: flush()
        if nrec%200000==0: print("recs",nrec,"states",len(seen_state),"used",len(used),flush=True)
flush()
print("TOTAL recs",nrec,"distinct states",len(seen_state))
# card DB text: catch cards that never appeared in the sampled selfplay
import csv
extra=set()
p="/root/ptcg/repo/data/JP_Card_Data.csv"
if os.path.exists(p):
    with open(p,newline="",encoding="utf-8") as f:
        rows=list(csv.reader(f))
    blob=[" ".join(x) for x in rows]
    for e in tok.encode_batch(blob): extra.update(e.ids)
    print("card-db adds", len(extra-used))
allids=used|extra
# safety: every single-character token
sc={i for t,i in tok.get_vocab().items() if len(t)<=1 or (t.startswith("Ġ") and len(t)<=2)}
print("single-char tokens", len(sc), "new", len(sc-allids))
allids|=sc
print("FINAL keep", len(allids), "of", V, "=", round(100*len(allids)/V,1),"%")
json.dump(sorted(int(x) for x in allids), open("/root/onnx/keep_ids.json","w"))
