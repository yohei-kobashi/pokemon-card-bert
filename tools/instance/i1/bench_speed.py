import time, warnings, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
M='Qwen/Qwen3.5-0.8B-Base'
tok=AutoTokenizer.from_pretrained(M)
if tok.pad_token is None: tok.pad_token=tok.eos_token
def build(grad_ckpt):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        m=AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map='auto')
        fp=[str(x.message) for x in w if 'fast path' in str(x.message).lower()]
    m.config.use_cache=False
    if grad_ckpt:
        m.gradient_checkpointing_enable(); m.enable_input_require_grads()
    m=get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules='all-linear', task_type='CAUSAL_LM'))
    for p in m.parameters():
        if p.requires_grad: p.data=p.data.float()
    return m, (len(fp)>0)
def bench(m, B, L=974, steps=15):
    from torch.optim import AdamW
    opt=AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4)
    ids=torch.randint(1000, 40000, (B, L), device='cuda')
    lab=ids.clone(); lab[:, :-8]=-100
    for _ in range(3):  # warmup
        opt.zero_grad(); m(input_ids=ids, labels=lab).loss.backward(); opt.step()
    torch.cuda.synchronize(); t=time.time()
    for _ in range(steps):
        opt.zero_grad(); m(input_ids=ids, labels=lab).loss.backward(); opt.step()
    torch.cuda.synchronize()
    dt=time.time()-t
    return B*steps/dt, torch.cuda.max_memory_allocated()/1e9
for name, gc, B in [('gradckpt_ON_b8', True, 8), ('gradckpt_OFF_b8', False, 8), ('gradckpt_OFF_b16', False, 16)]:
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        m, fallback = build(gc)
        sps, mem = bench(m, B)
        print(f'{name}: {sps:.2f} samp/s  peak {mem:.1f}GB  fla_fastpath={"OFF(fallback)" if fallback else "ON"}')
        del m
    except Exception as e:
        print(f'{name}: FAIL {type(e).__name__} {str(e)[:100]}')
        import gc as _g; _g.collect(); torch.cuda.empty_cache()
