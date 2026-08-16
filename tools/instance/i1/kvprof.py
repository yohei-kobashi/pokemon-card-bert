import os, time, math, ctypes, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import llama_cpp
from llama_cpp import Llama

llm = Llama(model_path="/root/sftv2.Q4_K_M.gguf", n_ctx=2048, n_threads=4,
            logits_all=False, verbose=False)
ctx = llm._ctx.ctx
NV = llm.n_vocab()
GZ = llama_cpp.llama_state_seq_get_size
G = llama_cpp.llama_state_seq_get_data
S = llama_cpp.llama_state_seq_set_data
GLI = llama_cpp.llama_get_logits_ith
GLI.restype = ctypes.POINTER(ctypes.c_float)


def tok(s):
    return llm.tokenize(s.encode(), add_bos=False, special=False)


def ll():
    p = GLI(ctx, -1)
    return np.ctypeslib.as_array(p, shape=(NV,)).astype(np.float64)


def lsm(l, i):
    a = l - l.max(); e = np.exp(a); return math.log(e[i] / e.sum())


prompt = ("[ACT]\nRULES " + " ".join("c%d card [SUP]: ability text here." % (i + 100) for i in range(20)) +
          "\nA[c345] || " + " ".join("%d=play c%d" % (i, i + 200) for i in range(12)))
cands = ["play c%d" % (i + 200) for i in range(12)]

pt = tok(prompt)
T = {}
t = time.time(); llm.reset(); llm.eval(pt); T["prompt_eval"] = time.time() - t
npr = llm.n_tokens
t = time.time(); pl = ll(); T["logit_read"] = time.time() - t
t = time.time(); sz = GZ(ctx, 0); buf = (ctypes.c_uint8 * sz)(); G(ctx, buf, sz, 0); T["seq_get"] = time.time() - t
T["state_MB"] = sz / 1e6

t_set = t_eval = t_lsm = 0.0
neval = 0
for cs in cands:
    ct = tok(cs) or [llm.token_eos()]
    a = time.time(); lp = lsm(pl, ct[0]); t_lsm += time.time() - a
    for k in range(1, len(ct)):
        a = time.time(); llm.eval([ct[k - 1]]); t_eval += time.time() - a; neval += 1
        a = time.time(); L = ll(); lp += lsm(L, ct[k]); t_lsm += time.time() - a
    a = time.time(); S(ctx, buf, sz, 0); llm.n_tokens = npr; t_set += time.time() - a
T["seq_set_total(12)"] = t_set
T["cand_eval_total(%d)" % neval] = t_eval
T["lsm_total"] = t_lsm
for k, v in T.items():
    print("%-22s %.4f" % (k, v), flush=True)
print("DONE", flush=True)
