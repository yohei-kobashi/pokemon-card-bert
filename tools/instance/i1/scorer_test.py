import os, time, math, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]=""
from llama_cpp import Llama
llm=Llama(model_path="/root/sftv2.Q4_K_M.gguf", n_ctx=4096, n_threads=4, logits_all=False, verbose=False)
def tok(s): return llm.tokenize(s.encode(), add_bos=False, special=False)
def last_logits(): return np.asarray(llm.scores[llm.n_tokens-1], dtype=np.float64)
def logsm_at(logits, idx):
    a=logits-logits.max(); e=np.exp(a); return math.log(e[idx]/e.sum())

def score(prompt, cand_strs):
    pt=tok(prompt)
    if len(pt) > 3900: pt=pt[-3900:]          # keep tail (menu+board), match maxlen budget
    llm.reset(); llm.eval(pt); n_prompt=llm.n_tokens
    prompt_last=last_logits()                 # predicts cand[0]
    out=[]
    for cs in cand_strs:
        ct=tok(cs) or [llm.token_eos()]
        lp=logsm_at(prompt_last, ct[0])
        for t in range(1,len(ct)):
            llm.eval([ct[t-1]])
            lp+=logsm_at(last_logits(), ct[t])
        # reset KV back to prompt
        llm._ctx.kv_cache_seq_rm(-1, n_prompt, -1)
        llm.n_tokens=n_prompt
        out.append(lp/len(ct))
    return out

# sanity: a plausible action vs gibberish
prompt="[ACT]\nRULES c1182 Boss’s Orders [SUP]: Switch in 1 of your opponent’s Benched Pokemon.\nA[c345] B[c344] pz4 dk50 || 0=attach c3 to c345 1=play c1182 2=end"
cands=["attach c3 to c345","play c1182","end","zzz random gibberish qqq"]
sc=score(prompt,cands)
print("sanity scores:", {c[:22]:round(s,3) for c,s in zip(cands,sc)})
print("argmax:", cands[int(np.argmax(sc))][:30], "(gibberish should NOT win)")
# determinism: same cand scored twice == same (KV reset correct)
s1=score(prompt,["play c1182","play c1182"]); print("KV-reset determinism (same cand twice):", round(s1[0],4), round(s1[1],4), "MATCH" if abs(s1[0]-s1[1])<1e-4 else "MISMATCH")
# timing on a realistic longer prompt (~900 tok) with 15 cands
import random; random.seed(1)
longp="[ACT]\nRULES "+" ".join(f"c{random.randint(1,1300)} card{i} [SUP]: some ability text here about the card." for i in range(20))+"\nA[c345] B[c344,c117] pz4 dk50 || "+" ".join(f"{i}=play c{random.randint(1,1300)}" for i in range(15))
lc=[f"play c{random.randint(1,1300)}" for _ in range(15)]
for _ in range(2): score(longp, lc)   # warm
t=time.time(); score(longp, lc); dt=time.time()-t
print(f"per-decision (prompt ~{len(tok(longp))} tok, 15 cands): {dt:.2f}s @4threads")
