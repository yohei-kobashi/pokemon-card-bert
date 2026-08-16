import os, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from llama_cpp import Llama

for nt in (4, 8, 16):
    llm = Llama(model_path="/root/sftv2.Q4_K_M.gguf", n_ctx=2048, n_threads=nt,
                n_threads_batch=nt, logits_all=False, verbose=False)
    toks = llm.tokenize(b"[ACT]\nRULES c1 card. c2 card. Board A[c3] || 0=end 1=play c9", add_bos=False)
    # prefill benchmark
    t = time.time(); llm.reset(); llm.eval(toks)
    pp = len(toks) / (time.time() - t)
    # tg benchmark: 32 single-token decodes
    last = toks[-1]; t = time.time()
    for _ in range(32):
        llm.eval([last]); last = int(__import__("numpy").argmax(__import__("numpy").ctypeslib.as_array(
            __import__("llama_cpp").llama_get_logits_ith(llm._ctx.ctx, -1), shape=(llm.n_vocab(),))))
    tg = 32 / (time.time() - t)
    print("nt=%2d  pp=%.1f t/s  tg=%.2f t/s" % (nt, pp, tg), flush=True)
    del llm
print("DONE", flush=True)
