# Verify llama-cpp-python low-level scoring API for KV-reuse candidate scoring
import os, time, math
os.environ["CUDA_VISIBLE_DEVICES"]=""
from llama_cpp import Llama
import llama_cpp, numpy as np
M="/root/sftv2.Q4_K_M.gguf"
llm=Llama(model_path=M, n_ctx=4096, n_threads=4, logits_all=False, verbose=False)
def tok(s): return llm.tokenize(s.encode("utf-8"), add_bos=False, special=False)

def logsm_at(logits, idx):
    a=np.asarray(logits, dtype=np.float64); a=a-a.max(); e=np.exp(a); return math.log(e[idx]/e.sum())

prompt="[ACT]\nRULES c1 test card\nA[c5] || 0=play 1=pass"
cands=["play","pass","use c1"]
pt=tok(prompt)
print("prompt tokens:", len(pt), "vocab:", llm.n_vocab())
# --- method: eval prompt once, then per candidate eval its tokens, read last-logits sequentially, reset KV ---
def score_kvreuse(pt, cand_strs):
    llm.reset(); llm.eval(pt); n_prompt=llm.n_tokens
    base=np.array(llm.scores[(n_prompt-1)*llm.n_vocab():(n_prompt)*llm.n_vocab()]) if hasattr(llm,"scores") else None
    # llama-cpp-python: after eval, llm.eval_logits or llm._scores? try the documented last-logits
    try:
        base=llm._scores[-1]   # shape [n_vocab]
    except Exception:
        base=np.array(llm.eval_logits[-1])
    out=[]
    for cs in cand_strs:
        ct=tok(cs) or [llm.token_eos()]
        lp=logsm_at(base, ct[0]); 
        for i in range(1,len(ct)):
            llm.eval([ct[i-1]]); 
            li=llm._scores[-1]
            lp+=logsm_at(li, ct[i])
        # reset KV to prompt
        llama_cpp.llama_kv_cache_seq_rm(llm.ctx, -1, n_prompt, -1)
        llm.n_tokens=n_prompt
        out.append(lp/len(ct))
    return out
t=time.time()
sc=score_kvreuse(pt, cands)
print("KVreuse scores:", [round(x,3) for x in sc], "in", round(time.time()-t,2),"s")
# reference: full eval prompt+cand each (create_completion echo logprobs) -- sanity
print("API attrs:", [a for a in dir(llm) if a in ("scores","_scores","eval_logits","ctx","n_tokens","eval","reset","n_vocab","token_eos")])
