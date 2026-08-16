"""BRANCH B, step 2 -- separate the three things a degraded gate can mean, on IDENTICAL decisions.

A gate says the new model plays worse. It cannot say why, and there are three candidates that
call for opposite responses:

  r8@full      d41_r8 on the prompt it was trained on          -- the reference
  r8@stripped  d41_r8 on the SAME decisions with DECK[] gone   -- did the FORMAT cost anything?
  s1@stripped  dusk_s1 on those same stripped decisions        -- did the RETRAIN cost anything?

r8@stripped ~ r8@full and s1 below both  =>  the retraining is at fault (lr, rows, l2sp).
r8@stripped well below r8@full           =>  the model did use DECK[], and removing it is the
                                             cost -- which contradicts `deck-segment-reliance`
                                             and would be worth knowing.
all three equal                          =>  ranking is intact and the loss is in PLAY, not in
                                             the ranking -- look at the per-opponent table.

The rows come from the UNSTRIPPED pool so both formats are derived from one source: every model
sees the same decisions, the same menus and the same labels, and the comparison is paired.
"""
import gzip, json, os, re, sys

ROOT = "/root/ptcg/repo"
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

SRC = os.path.join(ROOT, "data/rerank/v41_dusk.jsonl.gz")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
R8 = "/root/out/d41_r8"
S1 = "/root/out/dusk_s1"
STRIP = re.compile(r"^DECK .*?(?=T\d+\.)", re.S)
OPPS = {"dragapult", "marnie_grimmsnarl", "alakazam_nz", "alakazam", "crustle_geco", "crustle",
        "ogerpon_mono", "dudunsparce_box", "cynthia_garchomp", "mega_lucario_tr", "slowking"}

# Take from the TAIL: the head of the pool is the old carve, whose opponents are mostly decks
# no arm is being judged on.
keep = []
with gzip.open(SRC, "rt") as f:
    for line in f:
        d = json.loads(line)
        if d.get("opp") in OPPS and len(d.get("candidates") or []) >= 2:
            keep.append(d)
            if len(keep) > 40000:
                keep = keep[-20000:]
rows = keep[-N:]
print("scoring %d decisions | mean %.2f candidates"
      % (len(rows), sum(len(r["candidates"]) for r in rows) / max(1, len(rows))), flush=True)

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

dev = "cuda" if torch.cuda.is_available() else "cpu"


def top1(model_dir, strip):
    tok = AutoTokenizer.from_pretrained(model_dir)
    tok.truncation_side = "left"          # the menu is last; a right cut deletes the options
    # fp32 for BOTH arms. d41_r8 is stored bf16 and dusk_s1 will be stored fp32; loading each in
    # its own dtype would put a precision difference inside a comparison that is supposed to
    # isolate training from format. bf16 alone crushes the logit gaps this task turns on -- the
    # candidates of a real decision differ by ~0.004 and the bf16 grid near 1.0 is 0.0078.
    m = AutoModelForSequenceClassification.from_pretrained(
        model_dir, dtype=torch.float32).to(dev).eval()
    hit = []
    with torch.no_grad():
        for r in rows:
            st = STRIP.sub("", r["state"], count=1) if strip else r["state"]
            enc = tok([st] * len(r["candidates"]), r["candidates"], return_tensors="pt",
                      padding=True, truncation=True, max_length=512).to(dev)
            s = m(**enc).logits.squeeze(-1)
            hit.append(1 if int(s.argmax()) == r["chosen"] else 0)
    del m
    torch.cuda.empty_cache()
    return hit


arms = [("r8@full", R8, False), ("r8@stripped", R8, True), ("s1@stripped", S1, True)]
res = {}
for name, d, strip in arms:
    if not os.path.exists(os.path.join(d, "model.safetensors")):
        print("%-14s MISSING %s" % (name, d), flush=True)
        continue
    res[name] = top1(d, strip)
    print("%-14s top1 %.1f%%" % (name, 100.0 * sum(res[name]) / len(res[name])), flush=True)

print()
base = res.get("r8@full")
for name in ("r8@stripped", "s1@stripped"):
    if base and name in res:
        d = [a - b for a, b in zip(res[name], base)]
        m = sum(d) / len(d)
        sd = (sum((x - m) ** 2 for x in d) / (len(d) - 1)) ** 0.5 if len(d) > 1 else 0.0
        se = 100.0 * sd / (len(d) ** 0.5)
        print("%-14s minus r8@full: %+6.2f pt  se %.2f  t %6.2f"
              % (name, 100 * m, se, (100 * m / se) if se else 0.0))
json.dump({k: sum(v) / len(v) for k, v in res.items()},
          open("/root/loop_dusk/diag_rank.json", "w"), indent=1)
