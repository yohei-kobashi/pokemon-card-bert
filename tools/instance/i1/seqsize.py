import os, time, ctypes
os.environ["CUDA_VISIBLE_DEVICES"]=""
import llama_cpp
from llama_cpp import Llama
seqfns=[a for a in dir(llama_cpp) if "state_seq" in a.lower()]
print("SEQ_FNS:", seqfns)
llm=Llama(model_path="/root/sftv2.Q4_K_M.gguf", n_ctx=1536, n_threads=4, logits_all=False, verbose=False)
def tok(s): return llm.tokenize(s.encode(), add_bos=False, special=False)
llm.reset(); llm.eval(tok("[ACT]\nRULES c1 test card here for state\nA[c5] B[c6] || 0=play 1=pass")); npr=llm.n_tokens
ctx=llm._ctx.ctx
gsz=getattr(llama_cpp,"llama_state_seq_get_size",None)
gdat=getattr(llama_cpp,"llama_state_seq_get_data",None)
sdat=getattr(llama_cpp,"llama_state_seq_set_data",None)
print("have get_size/get_data/set_data:", bool(gsz),bool(gdat),bool(sdat))
if gsz:
    sz=gsz(ctx,0); print(f"seq-state size: {sz/1e6:.2f} MB")
    buf=(ctypes.c_uint8*sz)()
    t=time.time(); n=gdat(ctx, buf, sz, 0); tsave=time.time()-t
    # advance state (eval a couple tokens), then restore
    llm.eval(tok("play stuff")); 
    t=time.time(); sdat(ctx, buf, sz, 0); trest=time.time()-t
    print(f"seq_get_data: {tsave*1000:.1f} ms ({n} bytes) | seq_set_data(restore): {trest*1000:.1f} ms")
    print("=> low-level seq-state save/restore is", "FAST (viable)" if max(tsave,trest)<0.05 else "slow")
