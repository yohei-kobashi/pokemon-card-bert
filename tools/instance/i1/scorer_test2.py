# robust scorer: logits_all=True, eval(prompt) once, batch-eval each candidate, read cand-position logits, reset KV
import os, time, math, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]=""
from llama_cpp import Llama
llm=Llama(model_path="/root/sftv2.Q4_K_M.gguf", n_ctx=4096, n_threads=4, logits_all=True, verbose=False)
def tok(s): return llm.tokenize(s.encode(), add_bos=False, special=False)
def lsm(logits, idx):
    a=np.asarray(logits,dtype=np.float64); a=a-a.max(); e=np.exp(a); return math.log(e[idx]/e.sum())
def score(prompt, cand_strs):
    pt=tok(prompt)
    if len(pt)>3800: pt=pt[-3800:]
    llm.reset(); llm.eval(pt); n_prompt=llm.n_tokens
    # logits at position n_prompt-1 predict cand token 0
    out=[]
    for cs in cand_strs:
        ct=tok(cs) or [llm.token_eos()]
        llm.eval(ct)                       # batch-eval candidate tokens (logits_all -> all positions)
        # position (n_prompt-1 + i) predicts ct[i]
        lp=0.0
        for i in range(len(ct)):
            logits=llm.scores[n_prompt-1+i]
            lp+=lsm(logits, ct[i])
        llm._ctx.kv_cache_seq_rm(-1, n_prompt, -1); llm.n_tokens=n_prompt
        out.append(lp/len(ct))
    return out
prompt="[ACT]\nRULES c1182 Boss’s Orders [SUP]: Switch in an opponent Benched Pokemon.\nA[c345] B[c344] pz4 dk50 || 0=attach c3 to c345 1=play c1182 2=end"
cands=["attach c3 to c345","play c1182","end","zzz gibberish qqq random"]
sc=score(prompt,cands)
print("sanity:", {c[:20]:round(s,3) for c,s in zip(cands,sc)}, "| argmax:", cands[int(np.argmax(sc))][:22])
s1=score(prompt,["play c1182","play c1182"]); print("KV-reset determinism:", round(s1[0],4),round(s1[1],4), "OK" if abs(s1[0]-s1[1])<1e-4 else "MISMATCH")
import random; random.seed(1)
longp="[ACT]\nRULES "+" ".join(f"c{random.randint(1,1300)} card{i} [SUP]: ability text about this card here." for i in range(20))+"\nA[c345] B[c344,c117] || "+" ".join(f"{i}=play c{random.randint(1,1300)}" for i in range(15))
lc=[f"play c{random.randint(1,1300)}" for _ in range(15)]
for _ in range(2): score(longp,lc)
t=time.time(); score(longp,lc); print(f"per-decision(prompt {len(tok(longp))}tok,15cand): {time.time()-t:.2f}s @4thr")
