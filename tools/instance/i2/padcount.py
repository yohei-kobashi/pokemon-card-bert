"""Count the padding TRL actually produces, instead of simulating it.

measure_lengths.py predicted a random batch order would spend 1.30x the useful tokens at batch 8.
The A/B run disagreed: identical total_flos and 1.9% wall clock. One of the two is wrong about
what the dataloader does, and only the dataloader can settle it -- so this walks the real batches
and adds up padded vs real tokens.
"""
import argparse, gzip, json, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="unsloth/Qwen3-4B-Base")
    ap.add_argument("--action-vocab", default="")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--bsz", type=int, default=8)
    ap.add_argument("--maxlen", type=int, default=896)
    a = ap.parse_args()

    import torch
    from datasets import Dataset
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
    from trl import SFTConfig, SFTTrainer
    for p in (".", "cg-lib", "tools/instance"):
        if p not in sys.path:
            sys.path.insert(0, p)
    from lm.vocab import special_tokens
    from lm.action_token import action_token
    from sft_teacher import option_texts, make_trainer_class

    tok = AutoTokenizer.from_pretrained(a.model)
    tk = getattr(tok, "tokenizer", tok)
    act = json.load(open(a.action_vocab))["tokens"] if a.action_vocab else []
    tk.add_tokens(list(special_tokens()) + act)

    P, C = [], []
    with gzip.open(a.data, "rt") as f:
        for line in f:
            d = json.loads(line)
            t = d.get("target")
            if not t:
                continue
            o = option_texts(d["prompt"])
            if not o or int(t) >= len(o):
                continue
            P.append(d["prompt"])
            C.append(action_token(o[int(t)]))
            if len(P) >= a.n:
                break
    ds = Dataset.from_dict({"prompt": P, "completion": C})

    cfg = AutoConfig.from_pretrained(a.model)
    cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size = 1, 64, 128
    cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim = 4, 2, 16
    cfg.vocab_size, cfg.tie_word_embeddings = len(tk), True
    model = AutoModelForCausalLM.from_config(cfg)

    for group in (False, True):
        args = SFTConfig(max_length=a.maxlen, per_device_train_batch_size=a.bsz,
                         gradient_accumulation_steps=4, completion_only_loss=True, packing=False,
                         output_dir="/tmp/padcount", report_to="none", dataset_num_proc=1,
                         bf16=False, seed=3407)
        Cls = make_trainer_class(SFTTrainer, None, torch, group=group) if group else SFTTrainer
        tr = Cls(model=model, train_dataset=ds, processing_class=tok, args=args)
        padded = real = nb = 0
        for b in tr.get_train_dataloader():
            padded += b["input_ids"].numel()
            real += int(b["attention_mask"].sum())
            nb += 1
        print("%-14s batches %4d | real %8d | padded %8d | ratio %.4f | padded tok/sample %.1f"
              % ("length-sorted" if group else "random order", nb, real, padded,
                 padded / real, padded / len(P)))


if __name__ == "__main__":
    main()
