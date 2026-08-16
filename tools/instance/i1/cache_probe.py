import os, sys, torch
os.environ["CUDA_VISIBLE_DEVICES"]="0"
ROOT=os.path.expanduser("~/ptcg/repo")
for p in (ROOT, ROOT+"/cg-lib", ROOT+"/tools"): sys.path.insert(0,p)
from transformers import AutoTokenizer, AutoModelForCausalLM
BASE=ROOT+"/out/rl/sft_merged"
import json
cfg=json.load(open(BASE+"/config.json"))
lt=cfg.get("layer_types"); print("layer_types:", lt)
tok=AutoTokenizer.from_pretrained(BASE)
model=AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda").eval()
ids=tok(("board state "*40), return_tensors="pt").input_ids.cuda()[:, :48]
with torch.no_grad():
    out=model(input_ids=ids, use_cache=True)
pkv=out.past_key_values
print("PKV type:", type(pkv).__name__, "module:", type(pkv).__module__)
print("public attrs:", [a for a in dir(pkv) if not a.startswith("_")])
# probe list-like attributes and their shapes
for attr in dir(pkv):
    if attr.startswith("_"): continue
    try: v=getattr(pkv,attr)
    except Exception: continue
    if callable(v): continue
    if isinstance(v,(list,tuple)):
        shapes=[]
        for e in v[:26]:
            if isinstance(e,torch.Tensor): shapes.append(tuple(e.shape))
            elif isinstance(e,(list,tuple)): shapes.append([tuple(x.shape) if isinstance(x,torch.Tensor) else type(x).__name__ for x in e])
            else: shapes.append(type(e).__name__)
        print(f"  .{attr}: list len={len(v)} elems={shapes}")
    elif isinstance(v,torch.Tensor):
        print(f"  .{attr}: tensor {tuple(v.shape)}")
    else:
        print(f"  .{attr}: {type(v).__name__} = {str(v)[:60]}")
# batch-expand helpers present?
print("has batch_repeat_interleave:", hasattr(pkv,"batch_repeat_interleave"))
print("has reorder_cache:", hasattr(pkv,"reorder_cache"))
print("has batch_select_indices:", hasattr(pkv,"batch_select_indices"))
