"""Fast CPU per-decision microbench: time raw_score on a realistic synthetic decision.
   PyTorch f32, CPU. Reports ms/decision at the given BENCH_THREADS. Not Kaggle HW, pre-quant."""
import os, sys, time, statistics
os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = os.path.expanduser("~/ptcg/repo")
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    sys.path.insert(0, p)
import torch
TH = int(os.environ.get("BENCH_THREADS", "4"))
torch.set_num_threads(TH)
from transformers import AutoTokenizer, AutoModelForCausalLM
BASE = os.path.join(ROOT, "out", "rl", "sft_merged")
tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token_id is None: tok.pad_token = tok.eos_token
t0=time.time()
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32).eval()
print(f"loaded f32 CPU in {time.time()-t0:.1f}s, threads={TH}", flush=True)
PAD = tok.pad_token_id

def score(prompt_ids, cand_ids):
    seqs=[prompt_ids+c for c in cand_ids]; L=max(len(s) for s in seqs)
    inp=torch.full((len(seqs),L),PAD,dtype=torch.long); att=torch.zeros((len(seqs),L),dtype=torch.long)
    for i,s in enumerate(seqs): inp[i,:len(s)]=torch.tensor(s); att[i,:len(s)]=1
    start=len(prompt_ids); bpos=[]; ppos=[]
    for i,c in enumerate(cand_ids):
        for t in range(start,start+len(c)): bpos.append(i); ppos.append(t-1)
    with torch.no_grad():
        h=model.model(input_ids=inp,attention_mask=att).last_hidden_state
        bt=torch.tensor(bpos); pt=torch.tensor(ppos)
        lp=torch.log_softmax(model.lm_head(h[bt,pt]).float(),-1)
        tgt=inp[bt,pt+1]; tok_lp=lp[torch.arange(len(bpos)),tgt]
    out=[];k=0
    for c in cand_ids: out.append(float(tok_lp[k:k+len(c)].sum())/max(1,len(c)));k+=len(c)
    return out

# realistic decision: ~700-token prompt, N candidates x ~10 tokens
import random; random.seed(0)
def mk(plen, ncand, clen):
    pid=[random.randint(1000,40000) for _ in range(plen)]
    cid=[[random.randint(1000,40000) for _ in range(clen)] for _ in range(ncand)]
    return pid, cid

for ncand in (6, 15, 30):
    pid,cid=mk(700, ncand, 10)
    for _ in range(2): score(pid,cid)          # warmup
    ts=[]
    for _ in range(6):
        t=time.perf_counter(); score(pid,cid); ts.append((time.perf_counter()-t)*1000)
    print(f"  prompt700 x {ncand:2d} cands: {statistics.mean(ts):.0f} ms/decision "
          f"(median {statistics.median(ts):.0f}, {statistics.mean(ts)/ncand:.0f} ms/cand)", flush=True)
