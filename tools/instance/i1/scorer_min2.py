import os, math, numpy as np, time
os.environ["CUDA_VISIBLE_DEVICES"]=""
from llama_cpp import Llama
llm=Llama(model_path="/root/sftv2.Q4_K_M.gguf", n_ctx=2048, n_threads=4, logits_all=False, verbose=False)
def tok(s): return llm.tokenize(s.encode(), add_bos=False, special=False)
def lastl(): return np.asarray(llm.eval_logits[-1], dtype=np.float64)   # FIX: last computed logits
def lsm(l,i):
    a=l-l.max(); e=np.exp(a); return math.log(e[i]/e.sum())
def score(prompt,cands):
    pt=tok(prompt)
    if len(pt)>1900: pt=pt[-1900:]
    llm.reset(); llm.eval(pt); pl=lastl(); st=llm.save_state()
    out=[]
    for cs in cands:
        ct=tok(cs) or [llm.token_eos()]; lp=lsm(pl,ct[0])
        for t in range(1,len(ct)): llm.eval([ct[t-1]]); lp+=lsm(lastl(),ct[t])
        llm.load_state(st); out.append(lp/len(ct))
    return out
prompt="[ACT]\nRULES c1182 Boss’s Orders [SUP]: Switch an opponent Benched Pokemon.\nA[c345] B[c344] pz4 || 0=attach c3 to c345 1=play c1182 2=end"
cands=["attach c3 to c345","play c1182","end","zzz gibberish qqq random"]
sc=score(prompt,cands)
print("sanity:", {c[:16]:round(s,3) for c,s in zip(cands,sc)})
print("argmax:", cands[int(np.argmax(sc))][:24], "| gibberish rank:", sorted(range(4),key=lambda i:-sc[i]).index(3)+1,"of 4")
s1=score(prompt,["play c1182","play c1182"]); print("determinism:", round(s1[0],4),round(s1[1],4), "OK" if abs(s1[0]-s1[1])<1e-4 else "BAD")
import random; random.seed(1)
lp_="[ACT]\nRULES "+" ".join(f"c{random.randint(1,1300)} card{i} [SUP]: ability text here." for i in range(20))+"\nA[c345] B[c344] || "+" ".join(f"{i}=play c{random.randint(1,1300)}" for i in range(15))
lc=[f"play c{random.randint(1,1300)}" for _ in range(15)]
score(lp_,lc)
t=time.time(); score(lp_,lc); print(f"per-decision({len(tok(lp_))}tok,15cand): {time.time()-t:.2f}s")
