"""Create a fresh (zero-init) LoRA adapter on a base and save it, so the RL loop always
has a valid adapter dir to start from (gate/smoke eval + round-1 resume). Zero-init B
means the adapter is an identity at start -> the merged SFT policy is unchanged until RL
updates it.

    python tools/init_adapter.py --base <id_or_dir> --out <dir> [--r 16]
"""
import argparse


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--r", type=int, default=16)
    args = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    m = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map="cpu")
    m = get_peft_model(m, LoraConfig(r=args.r, lora_alpha=2 * args.r, lora_dropout=0.0,
                                     bias="none", target_modules="all-linear",
                                     task_type="CAUSAL_LM"))
    m.save_pretrained(args.out)          # LoRA B is zero-init -> identity adapter
    print(f"fresh adapter (r={args.r}) -> {args.out}")


if __name__ == "__main__":
    main()
