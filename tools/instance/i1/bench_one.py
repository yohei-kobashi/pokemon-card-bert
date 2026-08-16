import sys, time, warnings, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
name=sys.argv[1]; gc=sys.argv[2]=='1'; B=int(sys.argv[3])
M='Qwen/Qwen3.5-0.8B-Base'
tok=AutoTokenizer.from_pretrained(M)
if tok.pad_token is None: tok.pad_token=tok.eos_token
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    m=AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map='auto')
    fp=[x for x in w if 'fast path' in str(x.message).lower()]
m.config.use_cache=False
if gc: m.gradient_checkpointing_enable(); m.enable_input_require_grads()
m=get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules='all-linear', task_type='CAUSAL_LM'))
for p in m.parameters():
    if p.requires_grad: p.data=p.data.float()
opt=AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4)
L=974; core=m.base_model.model
def step():
    ids=torch.randint(1000,40000,(B,L),device='cuda'); att=torch.ones(B,L,dtype=torch.long,device='cuda')
    # supervised: last 8 positions of each row (matches _batch_loss efficient path)
    bpos=torch.arange(B,device='cuda').repeat_interleave(8)
    tpos=torch.cat([torch.arange(L-9,L-1,device='cuda') for _ in range(B)])
    tgt=torch.randint(1000,40000,(B*8,),device='cuda')
    opt.zero_grad()
    h=core.model(input_ids=ids, attention_mask=att).last_hidden_state
    hs=h[bpos,tpos]; logits=core.lm_head(hs).float()
    loss=F.cross_entropy(logits,tgt); loss.backward(); opt.step()
for _ in range(3): step()
torch.cuda.synchronize(); t=time.time()
for _ in range(20): step()
torch.cuda.synchronize(); dt=time.time()-t
print(f'RESULT {name}: {B*20/dt:.2f} samp/s  peak {torch.cuda.max_memory_allocated()/1e9:.1f}GB  fla_fastpath={"OFF" if fp else "ON"}')
