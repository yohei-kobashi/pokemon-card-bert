#!/usr/bin/env python3
"""Full divergence dump for weakness analysis: every human decision scored by the champion,
with menu context, both picks, the full menu, and the champion margin."""
import gzip, json, re, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
CH = "/root/out/fld_r49b"
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(CH)
model = AutoModelForSequenceClassification.from_pretrained(CH, torch_dtype=torch.float32).to(dev).eval()
rows = [json.loads(l) for l in gzip.open("/root/human_rows.jsonl.gz", "rt")]
out = gzip.open("/root/human_diverge.jsonl.gz", "wt", encoding="utf-8")
with torch.no_grad():
    for r in rows:
        hi = r.get("hi")
        if hi is None or not (0 <= hi < len(r["cands"])):
            continue
        enc = tok([r["prompt"]] * len(r["cands"]), r["cands"], return_tensors="pt",
                  padding=True, truncation=True, max_length=512).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            lg = model(**enc).logits.squeeze(-1).float()
        ci = int(lg.argmax())
        m = re.search(r"SEL (\S+)", r["prompt"])
        out.write(json.dumps({
            "opp": r.get("opp"), "t": r.get("t"), "won": r.get("won"),
            "ctx": m.group(1) if m else "?",
            "human": r["cands"][hi], "champ": r["cands"][ci],
            "agree": ci == hi,
            "margin": round(float(lg[ci] - lg[hi]), 3),
            "cands": r["cands"],
        }, ensure_ascii=False) + "\n")
out.close()
print("dumped")
