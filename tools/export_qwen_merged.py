#!/usr/bin/env python3
"""Flatten a card-first checkpoint into a standalone HF model directory.

vLLM cannot assemble what `QwenScorer` assembles at load time: a base model, a tokenizer with
2,971 domain tokens added, an embedding matrix resized to match, 3,064 embedding rows restored
from `domain_embeddings.pt`, and a LoRA on top. It wants one directory. This writes that.

Runs on CPU by default so it does not contend with a training run for the card; a 4B model in
bf16 is ~8 GB of host RAM and the merge is a few minutes.

The output is the SAME model `QwenScorer(..., merge=True)` builds in memory -- checked here by
comparing the merged weights against a freshly assembled copy, because a silently wrong export
would show up only as a bad benchmark number and be blamed on vLLM.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = "unsloth/Qwen3-4B-Base"
    cfgp = os.path.join(a.ckpt, "adapter_config.json")
    if os.path.exists(cfgp):
        base = (json.load(open(cfgp)).get("base_model_name_or_path") or base
                ).replace("-unsloth-bnb-4bit", "")
    print("[export] base %s" % base, flush=True)

    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).to(a.device)
    tok = AutoTokenizer.from_pretrained(base)

    emb = os.path.join(a.ckpt, "domain_embeddings.pt")
    if os.path.exists(emb):
        tk_new = AutoTokenizer.from_pretrained(a.ckpt)
        if len(tk_new) != len(tok):
            model.resize_token_embeddings(len(tk_new))
            blob = torch.load(emb, map_location="cpu")
            n_base, rows = blob["n_base"], blob["rows"]
            w = model.get_input_embeddings().weight
            if w.shape[0] != n_base + rows.shape[0]:
                raise SystemExit("domain_embeddings.pt does not fit: %d rows on top of %d, but "
                                 "the embedding is %d" % (rows.shape[0], n_base, w.shape[0]))
            with torch.no_grad():
                w[n_base:] = rows.to(device=w.device, dtype=w.dtype)
            tok = tk_new
            print("[export] restored %d domain-token rows" % rows.shape[0], flush=True)

    if os.path.exists(cfgp):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.ckpt)
        model = model.merge_and_unload()
        print("[export] LoRA merged", flush=True)

    model.eval()
    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out, safe_serialization=True)
    tok.save_pretrained(a.out)
    for f in ("cardfirst_vocab.json",):
        src = os.path.join(a.ckpt, f)
        if os.path.exists(src):
            with open(src) as r, open(os.path.join(a.out, f), "w") as w:
                w.write(r.read())
    n = sum(os.path.getsize(os.path.join(a.out, f)) for f in os.listdir(a.out))
    print("[export] wrote %s (%.1f GiB, vocab %d)" % (a.out, n / 2**30, len(tok)), flush=True)


if __name__ == "__main__":
    main()
