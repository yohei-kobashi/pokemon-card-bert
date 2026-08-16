import sys, torch, transformers
sys.path.insert(0,"/root/ptcg/repo"); sys.path.insert(0,"/root/ptcg/repo/cg-lib")
from transformers import AutoModelForSequenceClassification, AutoTokenizer
print("transformers", transformers.__version__)
d="/root/out/rerank_gte_none"
tok=AutoTokenizer.from_pretrained(d)
m=AutoModelForSequenceClassification.from_pretrained(d,trust_remote_code=True,dtype=torch.float32)
E=m.get_input_embeddings().weight.detach().float()
added=tok.get_added_vocab()
ids=sorted(v for v in added.values())
print("vocab_size(base)",tok.vocab_size,"added",len(added),"E",tuple(E.shape))
print("added id range",ids[0],"-",ids[-1])
base=E[:tok.vocab_size]; new=E[ids]
for nm,M in (("BASE rows",base),("ADDED rows",new)):
    mu=M.mean(0); res=M-mu
    print(f"  {nm:11s} |mu| {mu.norm():.4f}   residual per-row |.| mean {res.norm(dim=1).mean():.4f}"
          f"   -> residual/mu = {res.norm(dim=1).mean()/mu.norm():.4f}")
# is the added-row mean equal to the base-row mean? (that is what mean_resizing does)
print("cos(mean_added, mean_base) =", torch.nn.functional.cosine_similarity(
      new.mean(0)[None], base.mean(0)[None]).item())
# sanity: are two specific card tokens actually distinct single tokens?
for t in ("c344","c1152","d_alakazam","a_combo"):
    print(f"  {t:12s} -> id {tok.convert_tokens_to_ids(t)}  len(encode)={len(tok.tokenize(t))}")
