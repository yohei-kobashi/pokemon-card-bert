"""Merge a LoRA adapter into its base and save the bf16 merged model.

Used at two points: (1) after SFT (bf16 LoRA) to make the merged base for Stage A/B RL;
(2) at the Stage B->C boundary — merge the Stage-B adapter so Stage C's QLoRA quantizes a
single merged base (NF4) and trains a FRESH adapter on it (docs/rl_design.md; the one
quantization "jump" lands where broad knowledge is being discarded for specialization).

    python tools/merge_adapter.py --base <id> --adapter <dir> --out <dir>
"""
import argparse


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import os
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    # The SFT adapter carries a DOMAIN-EXTENDED vocab (base 248320 + ~2728 c<id>/a<id> tokens
    # -> 251048) and its trained rows live OUTSIDE the LoRA in new_embeddings.pt. Load the
    # adapter's tokenizer, resize the base embed/head to match, and restore those rows BEFORE
    # PeftModel.from_pretrained -- else the adapter's [251048,H] embed/lm_head shape-mismatches
    # the base's [248320,H]. (Same setup as sft_train_eval's --skip-train eval path.)
    tok_src = args.adapter if os.path.exists(os.path.join(args.adapter, "tokenizer_config.json")) else args.base
    tok = AutoTokenizer.from_pretrained(tok_src)
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16,
                                                 device_map="cpu")
    if len(tok) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tok), mean_resizing=False)
    ne = os.path.join(args.adapter, "new_embeddings.pt")
    if os.path.exists(ne):
        d = torch.load(ne, map_location="cpu")
        model.get_input_embeddings().weight.data[d["new_ids"]] = d["rows"].to(torch.bfloat16)
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()          # fold LoRA into the base weights
    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    print(f"merged {args.adapter} into {args.base} -> {args.out}")


if __name__ == "__main__":
    main()
