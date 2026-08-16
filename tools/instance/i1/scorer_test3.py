import os, time, math, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]=""
from llama_cpp import Llama
llm=Llama(model_path="/root/sftv2.Q4_K_M.gguf", n_ctx=2048, n_threads=4, logits_all=False, verbose=False)
def tok(s): return llm.tokenize(s.encode(), add_bos=False, special=False)
def lastl(): return np.asarray(llm.scores[llm.n_tokens-1], dtype=np.float64)
def lsm(logits, idx):
    a=logits-logits.max(); e=np.exp(a); return math.log(e[idx]/e.sum())
def score(prompt, cand_strs):
    pt=tok(prompt)
    if len(pt)>1900: pt=pt[-1900:]
    llm.reset(); llm.eval(pt); 
    pl=lastl()                       # predicts cand[0]
    st=llm.save_state()              # post-prompt state (recurrent + attn) -- correct branch point
    out=[]
    for cs in cand_strs:
        ct=tok(cs) or [llm.token_eos()]
        lp=lsm(pl, ct[0])
        for t in range(1,len(ct)):
            llm.eval([ct[t-1]]); lp+=lsm(lastl(), ct[t])
        llm.load_state(st)           # restore -> next candidate branches fresh from prompt
        out.append(lp/len(ct))
    return out
prompt="[ACT]\nRULES c1182 Boss’s Orders [SUP]: Switch an opponent Benched Pokemon.\nA[c345] B[c344] pz4 || 0=attach c3 to c345 1=play c1182 2=end"
cands=["attach c3 to c345","play c1182","end","zzz gibberish qqq"]
t=time.time(); sc=score(prompt,cands); 
print("sanity:", {c[:18]:round(s,3) for c,s in zip(cands,sc)}, "argmax:", cands[int(np.argmax(sc))][:20])
s1=score(prompt,["play c1182","play c1182"]); print("determinism:", round(s1[0],4),round(s1[1],4), "OK" if abs(s1[0]-s1[1])<1e-4 else "MISMATCH")
import random; random.seed(1)
longp="[ACT]\nRULES "+" ".join(f"c{random.randint(1,1300)} card{i} [SUP]: ability text about this card here." for i in range(20))+"\nA[c345] B[c344,c117] || "+" ".join(f"{i}=play c{random.randint(1,1300)}" for i in range(15))
lc=[f"play c{random.randint(1,1300)}" for _ in range(15)]
for _ in range(2): score(longp,lc)
tt=time.time(); score(longp,lc); print(f"per-decision(prompt {len(tok(longp))}tok,15cand): {time.time()-tt:.2f}s @4thr")
# state save/load overhead
t2=time.time(); 
for _ in range(15): llm.load_state(llm.save_state())
print(f"save+load_state x15: {time.time()-t2:.2f}s")
