import os, math, numpy as np, traceback
os.environ["CUDA_VISIBLE_DEVICES"]=""
from llama_cpp import Llama
llm=Llama(model_path="/root/sftv2.Q4_K_M.gguf", n_ctx=2048, n_threads=4, logits_all=False, verbose=False)
def tok(s): return llm.tokenize(s.encode(), add_bos=False, special=False)
def lastl(): return np.asarray(llm.scores[llm.n_tokens-1], dtype=np.float64)
def lsm(l,i):
    a=l-l.max(); e=np.exp(a); return math.log(e[i]/e.sum())
try:
    pt=tok("[ACT]\nRULES c1 test\nA[c5] || 0=play 1=pass")
    llm.reset(); llm.eval(pt); pl=lastl()
    print("STEP1 prompt eval ok, n=",llm.n_tokens)
    st=llm.save_state(); print("STEP2 save_state ok, type=",type(st).__name__)
    ct=tok("play")
    lp=lsm(pl,ct[0])
    for t in range(1,len(ct)):
        llm.eval([ct[t-1]]); lp+=lsm(lastl(),ct[t])
    print("STEP3 cand eval ok, lp=",round(lp/len(ct),3))
    llm.load_state(st); print("STEP4 load_state ok, n=",llm.n_tokens)
    ct2=tok("pass"); lp2=lsm(pl,ct2[0])
    for t in range(1,len(ct2)): llm.eval([ct2[t-1]]); lp2+=lsm(lastl(),ct2[t])
    print("STEP5 2nd cand ok, lp=",round(lp2/len(ct2),3))
    print("RESULT play vs pass:", round(lp/len(ct),3), round(lp2/len(ct2),3))
except Exception as e:
    print("FAILED:"); traceback.print_exc()
