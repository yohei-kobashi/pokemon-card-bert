import os, math, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]=""
from llama_cpp import Llama
import llama_cpp
llm=Llama(model_path="/root/sftv2.Q4_K_M.gguf", n_ctx=2048, n_threads=4, logits_all=False, verbose=False)
print("=== Llama attrs (logits/scores/ctx/kv) ===")
print([a for a in dir(llm) if any(k in a.lower() for k in ("score","logit","ctx","kv","n_tokens","eval","reset"))])
print("=== _ctx attrs (kv methods) ===")
c=getattr(llm,"_ctx",None)
print([a for a in dir(c) if "kv" in a.lower() or "seq" in a.lower() or "rm" in a.lower()] if c else "no _ctx")
print("=== llama_cpp module kv funcs ===")
print([a for a in dir(llama_cpp) if "kv" in a.lower() and ("rm" in a.lower() or "seq" in a.lower() or "clear" in a.lower())][:10])
# test eval + last logits
def tok(s): return llm.tokenize(s.encode(), add_bos=False, special=False)
llm.reset(); llm.eval(tok("[ACT]\ntest")); 
print("=== after eval: n_tokens ===", llm.n_tokens)
sc=llm.scores  # numpy?
print("llm.scores type/shape:", type(sc).__name__, getattr(sc,"shape",None))
try:
    last=llm._scores[-1]; print("_scores[-1] shape:", np.asarray(last).shape)
except Exception as e: print("_scores err:", e)
# does eval_logits exist?
print("has eval_logits:", hasattr(llm,"eval_logits"))
