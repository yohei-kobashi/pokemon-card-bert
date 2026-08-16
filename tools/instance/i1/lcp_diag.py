import os, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]=""
from llama_cpp import Llama
llm=Llama(model_path="/root/sftv2.Q4_K_M.gguf", n_ctx=2048, n_threads=4, logits_all=False, verbose=False)
def tok(s): return llm.tokenize(s.encode(), add_bos=False, special=False)
p=tok("[ACT]\nRULES c1 test\nA[c5] || 0=play 1=pass")
print("1) eval(prompt) len",len(p))
llm.reset(); llm.eval(p); print("   OK n_tokens=",llm.n_tokens)
print("2) eval(one more token) — does incremental eval work?")
try: llm.eval([p[0]]); print("   OK n_tokens=",llm.n_tokens)
except Exception as e: print("   FAIL:",repr(e)[:80])
print("3) save_state / load_state available?")
print("   has save_state:",hasattr(llm,"save_state"),"| load_state:",hasattr(llm,"load_state"))
print("4) save after prompt, then branch a candidate via load_state")
try:
    llm.reset(); llm.eval(p); npr=llm.n_tokens
    st=llm.save_state()
    llm.eval(tok("play")); print("   eval cand after save OK n_tokens=",llm.n_tokens)
    llm.load_state(st); print("   load_state OK, n_tokens restored=",llm.n_tokens, "(should be",npr,")")
    llm.eval(tok("pass")); print("   eval 2nd cand after restore OK n_tokens=",llm.n_tokens)
except Exception as e:
    import traceback; print("   FAIL:",repr(e)[:100])
