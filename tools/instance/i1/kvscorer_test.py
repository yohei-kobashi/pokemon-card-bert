import os, time, math, ctypes, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]=""
import llama_cpp
from llama_cpp import Llama
llm=Llama(model_path="/root/sftv2.Q4_K_M.gguf", n_ctx=2048, n_threads=4, logits_all=False, verbose=False)
ctx=llm._ctx.ctx
GET_SZ=llama_cpp.llama_state_seq_get_size; GET=llama_cpp.llama_state_seq_get_data; SET=llama_cpp.llama_state_seq_set_data
def tok(s): return llm.tokenize(s.encode(), add_bos=False, special=False)
def lastl(): return np.asarray(llm.eval_logits[-1], dtype=np.float64)
def lsm(l,i):
    a=l-l.max(); e=np.exp(a); return math.log(e[i]/e.sum())
def score(prompt, cands, maxlen=1900):
    pt=tok(prompt)
    if len(pt)>maxlen: pt=pt[-maxlen:]
    llm.reset(); llm.eval(pt); n_prompt=llm.n_tokens
    pl=lastl()
    sz=GET_SZ(ctx,0); buf=(ctypes.c_uint8*sz)(); GET(ctx,buf,sz,0)   # save post-prompt seq state
    out=[]
    for cs in cands:
        ct=tok(cs) or [llm.token_eos()]; lp=lsm(pl,ct[0])
        for t in range(1,len(ct)): llm.eval([ct[t-1]]); lp+=lsm(lastl(),ct[t])
        SET(ctx,buf,sz,0); llm.n_tokens=n_prompt                    # restore (low-level, ~10ms)
        out.append(lp/len(ct))
    return out
prompt="[ACT]\nRULES c1182 Boss’s Orders [SUP]: Switch an opponent Benched Pokemon.\nA[c345] B[c344] pz4 || 0=attach c3 to c345 1=play c1182 2=end"
cands=["attach c3 to c345","play c1182","end","zzz gibberish qqq random"]
sc=score(prompt,cands)
print("sanity:", {c[:16]:round(s,2) for c,s in zip(cands,sc)}, "argmax:", cands[int(np.argmax(sc))][:20])
print("gibberish is worst:", int(np.argmax([-x for x in sc]))==3)
s1=score(prompt,["play c1182","play c1182"]); print("determinism:", round(s1[0],4),round(s1[1],4), "OK" if abs(s1[0]-s1[1])<1e-4 else "MISMATCH")
import random; random.seed(1)
lp_="[ACT]\nRULES "+" ".join(f"c{random.randint(1,1300)} card{i} [SUP]: ability text about this card here." for i in range(20))+"\nA[c345] B[c344] || "+" ".join(f"{i}=play c{random.randint(1,1300)}" for i in range(15))
lc=[f"play c{random.randint(1,1300)}" for _ in range(15)]
score(lp_,lc)
t=time.time(); score(lp_,lc); print(f"per-decision(prompt {len(tok(lp_))}tok, 15 cand): {time.time()-t:.2f}s @4thr")
