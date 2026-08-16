"""Does the score depend on the CANDIDATE at all?

An eval loss frozen at 0.7397 across three learning rates, with the gradient shrinking as the
rate rises, is what a model looks like when its candidates receive identical logits: the
softmax is uniform, the target cannot be approached, and the gradient dies as the shared logit
saturates. That is a claim about the forward pass, so read the forward pass.

Three probes, each cheap and each conclusive on its own:
  1. the actual logits for a real row -- are they distinct?
  2. the same prompt against deliberately absurd candidates -- does the score move?
  3. the same candidate against two different prompts -- does the PROMPT move it?
"""
import gzip, json, sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DATA, MODEL = sys.argv[1], sys.argv[2]
rows = [json.loads(x) for x in gzip.open(DATA, "rt")][:400]
rows = [r for r in rows if len(r["cands"]) >= 2][:3]

dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL)
tok.truncation_side = "left"
m = AutoModelForSequenceClassification.from_pretrained(MODEL).to(dev).eval()
print("num_labels=%s  problem_type=%s" % (m.config.num_labels, getattr(m.config, "problem_type", None)))


def sc(prompt, cands):
    enc = tok([prompt] * len(cands), cands, return_tensors="pt", padding=True,
              truncation=True, max_length=512).to(dev)
    with torch.no_grad():
        out = m(**enc).logits
    return out.squeeze(-1).tolist(), enc["input_ids"].shape


for r in rows:
    s, shape = sc(r["prompt"], r["cands"])
    print("\nrow wc=%s\n  cands=%s\n  logits=%s  input_ids%s"
          % (r["wc"], r["cands"], ["%.4f" % x for x in s], tuple(shape)))

r = rows[0]
absurd = [r["cands"][0], "zzzz nonsense zzzz", "end", "attack:a999"]
s, _ = sc(r["prompt"], absurd)
print("\nsame prompt, absurd candidates:\n  %s\n  logits=%s"
      % (absurd, ["%.4f" % x for x in s]))

other = rows[1]["prompt"] if len(rows) > 1 else r["prompt"][::-1]
s1, _ = sc(r["prompt"], [r["cands"][0]])
s2, _ = sc(other, [r["cands"][0]])
print("\nsame candidate %r under two prompts: %.4f vs %.4f" % (r["cands"][0], s1[0], s2[0]))

# Do the two segments actually both reach the model? token_type_ids separates them; if the
# tokenizer is dropping the pair, every row's input is the prompt alone and the candidate is
# invisible no matter how long it trains.
enc = tok([r["prompt"]] * 2, r["cands"][:2], return_tensors="pt", padding=True,
          truncation=True, max_length=512)
ids = enc["input_ids"]
print("\ninput_ids differ between the two candidates: %s" % bool((ids[0] != ids[1]).any()))
print("first differing position: %s of %d tokens"
      % (int((ids[0] != ids[1]).nonzero()[0]) if bool((ids[0] != ids[1]).any()) else "-",
         ids.shape[1]))
print("tail of candidate 0: %r" % tok.decode(ids[0][-12:]))
print("tail of candidate 1: %r" % tok.decode(ids[1][-12:]))
