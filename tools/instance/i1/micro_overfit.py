"""Can the trainer memorise EIGHT rows? If not, nothing above it means anything.

The single-rule probe leaves the loss flat at two learning rates a decade apart, on data with
no duplicates, no contradictions, a median of two candidates, and a correct answer that is a
pure function of the menu text. That is the easiest supervised problem this codebase contains.
A 184M cross-encoder that cannot drive eight such rows to zero is not being asked a hard
question -- it is not being trained.

Prints the gradient norm alongside the loss, because "the loss does not move" has two very
different causes -- no gradient reaching the weights, and a gradient that is fighting itself --
and they are told apart by that number, not by the loss.
"""
import gzip, json, random, sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DATA = sys.argv[1]
MODEL = sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 8
LR = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-5

rows = [json.loads(x) for x in gzip.open(DATA, "rt")]
random.Random(0).shuffle(rows)
rows = [r for r in rows if len(r["cands"]) >= 2 and sum(r["wc"]) > 0][:N]
print("%d rows | cands %s" % (len(rows), [len(r["cands"]) for r in rows]))

dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL)
tok.truncation_side = "left"
DT = {"bf16": torch.bfloat16, "fp32": torch.float32}[sys.argv[5] if len(sys.argv) > 5 else "bf16"]
m = AutoModelForSequenceClassification.from_pretrained(MODEL, dtype=DT).to(dev)
print("param dtype: %s" % next(m.parameters()).dtype)
m.train()
opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.0)


def step_loss(r, do_step):
    enc = tok([r["prompt"]] * len(r["cands"]), r["cands"], return_tensors="pt",
              padding=True, truncation=True, max_length=512).to(dev)
    s = m(**enc).logits.squeeze(-1)
    lp = torch.log_softmax(s, dim=-1)
    t = torch.tensor(r["wc"], device=dev, dtype=lp.dtype)
    t = t / t.sum()
    loss = -(t * lp).sum()
    gn = float("nan")
    if do_step:
        opt.zero_grad()
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(m.parameters(), 1e9))
        opt.step()
    return float(loss), gn, int(s.argmax()), max(range(len(r["wc"])), key=lambda j: r["wc"][j])


def measure():
    """In eval() mode. Dropout is on during training, so the training-mode loss of a fully
    memorised row does not go to zero -- reading progress off it would confuse 'not learned'
    with 'measured through noise'."""
    m.eval()
    tot = hit = 0.0
    with torch.no_grad():
        for r in rows:
            l, _g, am, best = step_loss(r, do_step=False)
            tot += l
            hit += (r["wc"][am] >= r["wc"][best])
    m.train()
    return tot / len(rows), int(hit)


for it in range(41):
    gns = []
    if it % 5 == 0:
        el, eh = measure()
        print("iter %3d  eval-loss %.4f  argmax-correct %d/%d" % (it, el, eh, len(rows)))
    for r in rows:
        _l, gn, _am, _b = step_loss(r, do_step=(it > 0))
        if gn == gn:
            gns.append(gn)
    if it % 5 == 0 and gns:
        print("          grad-norm %.3e" % (sum(gns) / len(gns)))

# The single number that decides it: did the weights move at all?
tot = 0.0
for n_, p_ in m.named_parameters():
    if p_.grad is not None:
        tot += float(p_.grad.abs().sum())
print("sum |grad| over all parameters after the last step: %.4e" % tot)
