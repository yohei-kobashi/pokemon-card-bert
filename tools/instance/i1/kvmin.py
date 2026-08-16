import os, time, math, ctypes, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import llama_cpp
from llama_cpp import Llama

t0 = time.time()
llm = Llama(model_path="/root/sftv2.Q4_K_M.gguf", n_ctx=2048, n_threads=4,
            logits_all=False, verbose=False)
print("LOADED %.1fs" % (time.time() - t0), flush=True)

ctx = llm._ctx.ctx
GZ = llama_cpp.llama_state_seq_get_size
G = llama_cpp.llama_state_seq_get_data
S = llama_cpp.llama_state_seq_set_data
NV = llm.n_vocab()
GLI = llama_cpp.llama_get_logits_ith
GLI.restype = ctypes.POINTER(ctypes.c_float)


def tok(s):
    return llm.tokenize(s.encode(), add_bos=False, special=False)


def ll():
    p = GLI(ctx, -1)                       # last-token logits pointer
    return np.ctypeslib.as_array(p, shape=(NV,)).astype(np.float64)


def lsm(l, i):
    a = l - l.max()
    e = np.exp(a)
    return math.log(e[i] / e.sum())


def score(prompt, cands):
    pt = tok(prompt)
    if len(pt) > 1900:
        pt = pt[-1900:]
    llm.reset()
    llm.eval(pt)
    npr = llm.n_tokens
    pl = ll()
    sz = GZ(ctx, 0)
    buf = (ctypes.c_uint8 * sz)()
    G(ctx, buf, sz, 0)
    out = []
    for cs in cands:
        ct = tok(cs) or [llm.token_eos()]
        lp = lsm(pl, ct[0])
        for t in range(1, len(ct)):
            llm.eval([ct[t - 1]])
            lp += lsm(ll(), ct[t])
        S(ctx, buf, sz, 0)
        llm.n_tokens = npr
        out.append(lp / len(ct))
    return out


p = "[ACT]\nRULES c1182 Boss Orders [SUP]: Switch an opponent Pokemon.\nA[c345] B[c344] || 0=attach c3 1=play c1182 2=end"
t = time.time()
sc = score(p, ["attach c3", "play c1182", "end", "zzz junk"])
print("SANITY %.2fs" % (time.time() - t), [round(x, 2) for x in sc],
      "gib_worst=%s" % (int(np.argmin(sc)) == 3), flush=True)

s1 = score(p, ["play c1182", "play c1182"])
print("DETERMINISM match=%s (%.4f vs %.4f)" % (round(s1[0], 4) == round(s1[1], 4), s1[0], s1[1]), flush=True)

lp_ = ("[ACT]\nRULES " + " ".join("c%d card [SUP]: ability text here." % (i + 100) for i in range(20)) +
       "\nA[c345] || " + " ".join("%d=play c%d" % (i, i + 200) for i in range(12)))
ntok = len(tok(lp_))
t = time.time()
score(lp_, ["play c%d" % (i + 200) for i in range(12)])
print("PERDECISION %.2fs ntok=%d ncand=12" % (time.time() - t, ntok), flush=True)
print("ALLDONE", flush=True)
