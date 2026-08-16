import os, sys, time, torch, statistics
os.environ["CUDA_VISIBLE_DEVICES"]="0"
ROOT=os.path.expanduser("~/ptcg/repo")
for p in (ROOT, ROOT+"/cg-lib", ROOT+"/tools"): sys.path.insert(0,p)
from transformers import AutoTokenizer, AutoModelForCausalLM
BASE=ROOT+"/out/rl/sft_merged"
tok=AutoTokenizer.from_pretrained(BASE)
if tok.pad_token_id is None: tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda").eval()
dev=model.device; PAD=tok.pad_token_id; core=model; MAXLEN=1024

def prep(prompt,cands):
    p=tok(prompt,add_special_tokens=False)["input_ids"]
    c=[tok(x,add_special_tokens=False)["input_ids"] or [tok.eos_token_id] for x in cands]
    cap=MAXLEN-max(len(z) for z in c)
    if cap>0 and len(p)>cap: p=p[-cap:]
    return p,c

def naive(prompt,cands):
    p,c=prep(prompt,cands); seqs=[p+z for z in c]; L=max(len(s) for s in seqs)
    inp=torch.full((len(seqs),L),PAD,dtype=torch.long,device=dev); att=torch.zeros((len(seqs),L),dtype=torch.long,device=dev)
    for i,s in enumerate(seqs): inp[i,:len(s)]=torch.tensor(s,device=dev); att[i,:len(s)]=1
    start=len(p); bpos=[];ppos=[]
    for i,z in enumerate(c):
        for t in range(start,start+len(z)): bpos.append(i);ppos.append(t-1)
    with torch.no_grad():
        h=core.model(input_ids=inp,attention_mask=att).last_hidden_state
        bt=torch.tensor(bpos,device=dev);pt=torch.tensor(ppos,device=dev)
        lp=torch.log_softmax(core.lm_head(h[bt,pt]).float(),-1); tgt=inp[bt,pt+1]
        tl=lp[torch.arange(len(bpos),device=dev),tgt]
    out=[];k=0
    for z in c: out.append(float(tl[k:k+len(z)].sum())/max(1,len(z)));k+=len(z)
    return out

def broadcast_cache(past,n):
    for L in past.layers:
        k=getattr(L,"keys",None)
        if k is not None:                                   # DynamicLayer (attention)
            L.keys=L.keys.repeat_interleave(n,dim=0).contiguous()
            L.values=L.values.repeat_interleave(n,dim=0).contiguous()
        else:                                               # LinearAttentionLayer
            for si in range(getattr(L,"number_of_states",1)):
                cs=getattr(L,"conv_states",None); rs=getattr(L,"recurrent_states",None)
                if cs is not None and cs[si] is not None: cs[si]=cs[si].repeat_interleave(n,dim=0).contiguous()
                if rs is not None and rs[si] is not None: rs[si]=rs[si].repeat_interleave(n,dim=0).contiguous()

def kvreuse(prompt,cands):
    p,c=prep(prompt,cands); n=len(c); Lp=len(p); P=torch.tensor([p],device=dev)
    with torch.no_grad():
        o1=core.model(input_ids=P,use_cache=True); past=o1.past_key_values
        h_last=o1.last_hidden_state[:,-1,:]
        lp0=torch.log_softmax(core.lm_head(h_last).float(),-1)[0]
        broadcast_cache(past,n)
        maxc=max(len(z) for z in c)
        cinp=torch.full((n,maxc),PAD,dtype=torch.long,device=dev)
        att=torch.zeros((n,Lp+maxc),dtype=torch.long,device=dev); att[:,:Lp]=1
        for i,z in enumerate(c): cinp[i,:len(z)]=torch.tensor(z,device=dev); att[i,Lp:Lp+len(z)]=1
        pos=torch.arange(Lp,Lp+maxc,device=dev).unsqueeze(0).expand(n,-1)
        o2=core.model(input_ids=cinp,past_key_values=past,position_ids=pos,attention_mask=att,use_cache=False)
        H=o2.last_hidden_state
    out=[]
    for i,z in enumerate(c):
        tot=lp0[z[0]]
        if len(z)>1:
            hi=H[i,:len(z)-1,:]; lpi=torch.log_softmax(core.lm_head(hi).float(),-1)
            idx=torch.tensor(z[1:],device=dev); tot=tot+lpi[torch.arange(len(z)-1,device=dev),idx].sum()
        out.append(float(tot)/max(1,len(z)))
    return out

import random; random.seed(1)
prompt="[ACT]\n"+" ".join(random.choice(["bench","active","energy","prize","hand","attach","retreat","attack","supporter","damage"]) for _ in range(650))
cands=[" ".join(random.choice(["play","use","attach","pass","retreat","bench"]) for _ in range(random.randint(3,12))) for _ in range(15)]
a=naive(prompt,cands); b=kvreuse(prompt,cands)
diffs=[abs(x-y) for x,y in zip(a,b)]
print("naive[:5] :",[round(x,4) for x in a[:5]]); print("kvreuse[:5]:",[round(x,4) for x in b[:5]])
print(f"MAX ABS DIFF: {max(diffs):.4e}  mean: {statistics.mean(diffs):.4e}")
ra=sorted(range(len(a)),key=lambda i:-a[i]); rb=sorted(range(len(b)),key=lambda i:-b[i])
print("argmax match:",a.index(max(a))==b.index(max(b)),"  ranking identical:",ra==rb)
for name,fn in (("naive",naive),("kvreuse",kvreuse)):
    fn(prompt,cands); torch.cuda.synchronize(); t=time.time()
    for _ in range(15): fn(prompt,cands)
    torch.cuda.synchronize(); print(f"{name}: {(time.time()-t)/15*1000:.1f} ms/decision")
