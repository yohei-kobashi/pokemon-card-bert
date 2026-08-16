"""Three hypotheses have failed (divergence=bug, attack labels, duplicate prompts). Stop
guessing: after the probe, look at the PER-ROW loss. If most rows are near their own floor and
a few are stuck, the residue is a subset to inspect. If every row is uniformly short, it is
capacity or an input the model cannot see."""
import gzip, json, math, random, sys, statistics as st
sys.path.insert(0, "/root/ptcg/repo"); sys.path.insert(0, "/root/ptcg/repo/cg-lib")
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

rows = [json.loads(x) for x in gzip.open("/root/rl/plan_r4.jsonl.gz", "rt")]
random.Random(0).shuffle(rows)
rows = rows[:300]
dev = "cuda"
tok = AutoTokenizer.from_pretrained("/root/out/d41_r8"); tok.truncation_side = "left"
m = AutoModelForSequenceClassification.from_pretrained("/root/out/d41_r8").to(dev)
m.train()
opt = torch.optim.AdamW(m.parameters(), lr=5e-6, weight_decay=0.0)

def fwd(r):
    enc = tok([r["prompt"]] * len(r["cands"]), r["cands"], return_tensors="pt",
              padding=True, truncation=True, max_length=512).to(dev)
    return m(**enc).logits.squeeze(-1)

ACC = 8
for ep in range(30):
    for i, r in enumerate(rows):
        lp = torch.log_softmax(fwd(r), -1)
        t = torch.tensor(r["wc"], device=dev, dtype=lp.dtype); t = t / t.sum()
        ((-(t * lp).sum()) / ACC).backward()
        if (i + 1) % ACC == 0:
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); opt.zero_grad()

m.eval()
res = []
with torch.no_grad():
    for r in rows:
        lp = torch.log_softmax(fwd(r), -1)
        t = torch.tensor(r["wc"], device=dev, dtype=lp.dtype); t = t / t.sum()
        loss = float(-(t * lp).sum())
        floor = float(-(t * torch.log(t.clamp_min(1e-9))).sum())
        best = max(range(len(r["wc"])), key=lambda j: r["wc"][j])
        res.append((loss - floor, loss, floor, int(lp.argmax()) == best or r["wc"][int(lp.argmax())] >= r["wc"][best], r))
res.sort(key=lambda x: -x[0])
gaps = [x[0] for x in res]
print("PER-ROW gap above own floor: mean %.3f | p50 %.3f | p90 %.3f | max %.3f"
      % (st.mean(gaps), gaps[len(gaps)//2], gaps[int(.1*len(gaps))], gaps[0]))
print("rows within 0.1 of their floor: %d/300 | argmax on a top-weight candidate: %d/300"
      % (sum(1 for g in gaps if g < 0.1), sum(1 for x in res if x[3])))
print("\nWORST 5 rows:")
for g, loss, fl, ok, r in res[:5]:
    print("  gap %.2f  rules=%s  cands=%d  wc=%s" % (g, r["rules"], len(r["cands"]), r["wc"]))
    print("      %s" % " | ".join(r["cands"][:6]))
