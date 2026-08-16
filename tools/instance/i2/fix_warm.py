#!/usr/bin/env python3
"""Teach warm_start the name a saved LoRA tensor goes by inside a live PEFT model."""
import pathlib
import sys

p = pathlib.Path(sys.argv[1])
s = p.read_text()

old = """            for k, v in sd.items():
                for c in (k, k.replace("base_model.model.", ""), "base_model.model." + k):
                    if c in cur and cur[c].shape == v.shape:"""
new = """            for k, v in sd.items():
                for c in _adapter_key_forms(k):
                    if c in cur and cur[c].shape == v.shape:"""
assert old in s, "matcher anchor not found"
s = s.replace(old, new)

helper = '''def _adapter_key_forms(k):
    """Every name a saved adapter tensor can go by inside a live PEFT model.

    PEFT keeps adapters under their NAME, so a live parameter is `...lora_A.default.weight`
    while `save_pretrained` writes `...lora_A.weight`. The three prefix variants tried before
    matched 0 of 504 tensors; the refusal fired correctly and the chain stopped for ten hours
    with an idle GPU. The gate was right, the matcher was wrong.
    """
    out = []
    for b in (k, k.replace("base_model.model.", ""), "base_model.model." + k):
        out.append(b)
        if b.endswith(".weight"):
            out.append(b[: -len(".weight")] + ".default.weight")
    return out


'''
anchor = "def warm_start(model, tk, torch, src, n_base):"
assert anchor in s, "warm_start anchor not found"
s = s.replace(anchor, helper + anchor, 1)
p.write_text(s)
print("patched:", p)
