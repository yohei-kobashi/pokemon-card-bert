import os, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from llama_cpp import Llama

BASE = "card [SUP]: ability text here and more words to fill. "


def run(nb, nub):
    llm = Llama(model_path="/root/sftv2.Q4_K_M.gguf", n_ctx=2048, n_threads=4,
                n_batch=nb, n_ubatch=nub, logits_all=False, verbose=False)
    print("=== n_batch=%d n_ubatch=%d ===" % (nb, nub), flush=True)
    for reps in (5, 25, 60):
        toks = llm.tokenize((("[ACT]\n" + BASE * reps)).encode(), add_bos=False)
        t = time.time(); llm.reset(); llm.eval(toks); dt = time.time() - t
        print("  ntok=%4d  %.2fs  pp=%.1f t/s" % (len(toks), dt, len(toks) / dt), flush=True)
    del llm


run(512, 512)
run(512, 128)
run(2048, 2048)
print("DONE", flush=True)
