#!/usr/bin/env python3
"""Champion-vs-human divergences -> heavily weighted DPO rows (user directive 2026-08-17).

Scores every human decision (/root/human_rows.jsonl.gz, `hi` = human pick) with the CHAMPION
cross-encoder; wherever the champion argmax differs, emits {prompt, cands: [human, champion],
wc: [0.95, 0.05]} x3 copies -- the strongest correction signal the games can give: the exact
states where the current policy would have played differently from the winning human.
"""
import gzip, json, sys, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

CH = "/root/out/fld_r49b"
MAXLEN = 512
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(CH)
model = AutoModelForSequenceClassification.from_pretrained(CH, torch_dtype=torch.float32).to(dev).eval()

rows = [json.loads(l) for l in gzip.open("/root/human_rows.jsonl.gz", "rt")]
n_div = n_agree = 0
with gzip.open("/root/human_dpo_rows.jsonl.gz", "wt", encoding="utf-8") as out:
    with torch.no_grad():
        for r in rows:
            hi = r.get("hi")
            if hi is None or not (0 <= hi < len(r["cands"])):
                continue
            enc = tok([r["prompt"]] * len(r["cands"]), r["cands"], return_tensors="pt",
                      padding=True, truncation=True, max_length=MAXLEN).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                logits = model(**enc).logits.squeeze(-1)
            ci = int(logits.float().argmax())
            if ci == hi:
                n_agree += 1
                continue
            n_div += 1
            row = json.dumps({"prompt": r["prompt"], "cands": [r["cands"][hi], r["cands"][ci]],
                              "wc": [0.95, 0.05], "src": "hdpo", "opp": r.get("opp"),
                              "won": r.get("won")}, ensure_ascii=False) + "\n"
            for _ in range(3):
                out.write(row)
print("[hdpo] %d decisions: champion agrees %d, diverges %d -> %d rows (x3)"
      % (len(rows), n_agree, n_div, n_div * 3))
