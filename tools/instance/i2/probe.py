import os, sys, torch
ROOT="/root/ptcg/repo"
for p in (ROOT, ROOT+"/cg-lib", ROOT+"/tools"):
    sys.path.insert(0,p)
from tools.mirror_match import QwenScorer
sc = QwenScorer("/root/out/qwen3_4b_cfb_v40", merge=False, kv=False, backend="hf")
m, tk, dev = sc.model, sc.tk, sc.model.device
ids = tk("[ACT]\nDECK win[c743x4] eng[c13] T3.2 ME A[c5:100/100] pz3 dk20 bm5 H[c7,c9] || SEL MAIN n1-1 :: 0=attach:c7@ACTIVE 1=end",
         add_special_tokens=False)["input_ids"]
print("prompt tokens", len(ids))
t = torch.tensor([ids], device=dev)
with torch.no_grad():
    full = m(input_ids=t, use_cache=False)
    print("plain logits shape", tuple(full.logits.shape))
    try:
        o = m(input_ids=t, use_cache=False, logits_to_keep=1)
        print("logits_to_keep shape", tuple(o.logits.shape),
              "maxdiff", float((o.logits[0,-1].float()-full.logits[0,-1].float()).abs().max()))
    except Exception as e:
        print("logits_to_keep RAISED:", type(e).__name__, e)
    o1 = m(input_ids=torch.tensor([ids[:-1]], device=dev), use_cache=True)
    print("cache type", type(o1.past_key_values).__name__ if o1.past_key_values is not None else None)
    if o1.past_key_values is not None:
        o2 = m(input_ids=torch.tensor([[ids[-1]]], device=dev),
               past_key_values=o1.past_key_values, use_cache=True)
        a = torch.log_softmax(o2.logits[0,-1].float(),-1)
        b = torch.log_softmax(full.logits[0,-1].float(),-1)
        print("kv logprob maxdiff", float((a-b).abs().max()),
              "argmax equal", int(a.argmax())==int(b.argmax()))
        print("has crop", hasattr(o1.past_key_values,"crop"))
