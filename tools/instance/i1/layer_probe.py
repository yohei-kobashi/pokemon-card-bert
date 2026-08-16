import os, sys, torch
os.environ["CUDA_VISIBLE_DEVICES"]="0"
ROOT=os.path.expanduser("~/ptcg/repo")
for p in (ROOT, ROOT+"/cg-lib", ROOT+"/tools"): sys.path.insert(0,p)
from transformers import AutoTokenizer, AutoModelForCausalLM
BASE=ROOT+"/out/rl/sft_merged"
tok=AutoTokenizer.from_pretrained(BASE); 
if tok.pad_token_id is None: tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda").eval()
ids=tok("board "*40, return_tensors="pt").input_ids.cuda()[:,:48]
with torch.no_grad(): past=model(input_ids=ids,use_cache=True).past_key_values
for li in (0,3):   # 0=LinearAttentionLayer, 3=DynamicLayer
    L=past.layers[li]
    print(f"=== layer {li}: {type(L).__name__} ===")
    for a in dir(L):
        if a.startswith("_"): continue
        try: v=getattr(L,a)
        except Exception: continue
        if callable(v): continue
        if isinstance(v,torch.Tensor): print(f"  .{a}: Tensor {tuple(v.shape)} {v.dtype}")
        elif isinstance(v,(list,tuple)) and v and isinstance(v[0],torch.Tensor): print(f"  .{a}: list[{len(v)}] Tensor {tuple(v[0].shape)}")
        elif isinstance(v,(int,float,bool,str,type(None))): print(f"  .{a}: {type(v).__name__}={v}")
    # methods of interest
    print("  methods:", [m for m in dir(L) if not m.startswith('_') and callable(getattr(L,m,None))])
