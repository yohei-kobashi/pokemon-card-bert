import sys, torch
sys.path.insert(0,"/root/ptcg/repo"); sys.path.insert(0,"/root/ptcg/repo/cg-lib")
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from lm import vocab
d="/root/out/_init_test"
tok=AutoTokenizer.from_pretrained(d)
m=AutoModelForSequenceClassification.from_pretrained(d,trust_remote_code=True,dtype=torch.float32)
E=m.get_input_embeddings().weight.detach().float()
names=[f"c{c}" for c in vocab._CARDS]
decks,arches=vocab._fleet_names()
names+= [vocab.deck_tok(x) for x in decks]
ids=[tok.convert_tokens_to_ids(t) for t in names]
M=E[ids]; M=M/M.norm(dim=1,keepdim=True)
def lab(t):
    if t.startswith("d_"): return t
    return f"{t} {vocab.card_name(int(t[1:]))}"
for q in ("c344","c1152","d_alakazam","d_crustle_stall","c19"):
    i=names.index(q); s=M@M[i]; s[i]=-9
    top=torch.topk(s,5).indices.tolist()
    print(f"{lab(q)}\n   -> " + "\n   -> ".join(f"{s[j]:+.3f} {lab(names[j])}" for j in top))
