"""Can the trainer memorise ONE row? If not, the loss/optimizer path is broken and no
hyperparameter will fix it. The floor for a single row is its own target entropy."""
import gzip, json, math, sys
sys.path.insert(0, "/root/ptcg/repo"); sys.path.insert(0, "/root/ptcg/repo/cg-lib")
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

rows = [json.loads(x) for x in gzip.open("/root/rl/plan_r4.jsonl.gz", "rt")]
r = next(x for x in rows if 2 <= len(x["cands"]) <= 5 and 0 < sum(1 for w in x["wc"] if w > 0) < len(x["cands"]))
t = [w / sum(r["wc"]) for w in r["wc"]]
floor = -sum(p * math.log(p) for p in t if p > 0)
print("row: %d cands | wc=%s | FLOOR %.4f" % (len(r["cands"]), r["wc"], floor))

dev = "cuda"
tok = AutoTokenizer.from_pretrained("/root/out/d41_r8"); tok.truncation_side = "left"
m = AutoModelForSequenceClassification.from_pretrained("/root/out/d41_r8").to(dev)
m.train()
opt = torch.optim.AdamW(m.parameters(), lr=1e-5, weight_decay=0.0)
tgt = torch.tensor(t, device=dev)
for step in range(301):
    enc = tok([r["prompt"]] * len(r["cands"]), r["cands"], return_tensors="pt",
              padding=True, truncation=True, max_length=512).to(dev)
    s = m(**enc).logits.squeeze(-1)
    lp = torch.log_softmax(s, dim=-1)
    loss = -(tgt * lp).sum()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step(); opt.zero_grad()
    if step % 50 == 0:
        print("  step %3d  loss %.4f  (floor %.4f)  scores=%s"
              % (step, float(loss), floor, [round(float(x), 2) for x in s]))
