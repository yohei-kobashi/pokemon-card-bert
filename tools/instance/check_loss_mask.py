#!/usr/bin/env python3
"""Prove that the SFT loss is taken on the ANSWER ONLY, not on the prompt.

`completion_only_loss=True` is set in the SFTConfig, but a flag being set is not evidence that it
fired -- this codebase has repeatedly shipped configuration that silently did nothing (a line
config naming a cut card, an archetype ladder overridden by l2, a `-DECK[]` ablation that removed
nothing). If the prompt were also being trained on, the run would still converge and still print
a falling loss: the model would simply spend most of its capacity learning to reproduce board
states, and only the win rate would ever show it.

So this drives TRL's OWN dataset processing and collator -- not a re-implementation of them, which
could drift from what actually trains -- and reports which token positions carry a label.

It runs on CPU with a 2-layer randomly-initialised model. Only the TOKENIZER and the SFTConfig
decide the masking, so a small model gives exactly the same answer as the 4B and leaves the GPU
free for the benchmark.

Expected: labels are -100 across the whole prompt, and live on the completion only -- for this
task that is a single digit (plus whatever end-of-sequence token TRL appends).
"""
import argparse
import gzip
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="unsloth/Qwen3-4B-Base")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--maxlen", type=int, default=896)
    ap.add_argument("--domain-tokens", action="store_true")
    ap.add_argument("--index-tokens", action="store_true")
    ap.add_argument("--action-vocab", default="")
    ap.add_argument("--card-first", default="")
    ap.add_argument("--keep-eos", action="store_true")
    a = ap.parse_args()

    import torch
    from datasets import Dataset
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(a.model)
    tk = getattr(tok, "tokenizer", tok)
    if a.domain_tokens:
        for p in ("/root/ptcg/repo", "/root/ptcg/repo/cg-lib", ".", "cg-lib"):
            if p not in sys.path:
                sys.path.insert(0, p)
        from lm.vocab import special_tokens
        sys.path.insert(0, "tools/instance")
        from sft_teacher import INDEX_TOKENS
        act = json.load(open(a.action_vocab))["tokens"] if a.action_vocab else []
        cf = []
        if a.card_first:
            _c = json.load(open(a.card_first))
            cf = list(_c["new_tokens"]) + list(_c.get("second_tokens")
                                                or _c.get("sub_tokens") or [])
        tk.add_tokens(list(special_tokens()) + (INDEX_TOKENS if a.index_tokens else [])
                      + act + cf)

    P, C = [], []
    with gzip.open(a.data, "rt") as f:
        for line in f:
            d = json.loads(line)
            if not d.get("target"):
                continue
            P.append(d["prompt"])
            if a.card_first:
                from sft_teacher import option_texts
                from lm.action_token import (first_token, sub_index, SUB_TOKENS,
                                             to_scheme_b, label_b)
                o = option_texts(d["prompt"])
                if not o or int(d["target"]) >= len(o):
                    P.pop()
                    continue
                if json.load(open(a.card_first)).get("scheme") == "b":
                    P[-1] = to_scheme_b(d["prompt"])
                    x_, y_ = label_b(d["prompt"], int(d["target"]), o)
                    C.append(x_ + (y_ or ""))
                else:
                    ft = first_token(o[int(d["target"])])
                    si = sub_index(d["prompt"], o, int(d["target"]))
                    C.append(ft if si is None else ft + SUB_TOKENS[si])
            elif a.action_vocab:
                from sft_teacher import option_texts
                from lm.action_token import action_token
                o = option_texts(d["prompt"])
                if not o or int(d["target"]) >= len(o):
                    continue
                C.append(action_token(o[int(d["target"])]))
            else:
                C.append(INDEX_TOKENS[int(d["target"])] if a.index_tokens else d["target"])
            if len(P) >= a.n:
                break
    ds = Dataset.from_dict({"prompt": P, "completion": C})

    cfg = AutoConfig.from_pretrained(a.model)
    cfg.num_hidden_layers = 2
    cfg.hidden_size = 64
    cfg.intermediate_size = 128
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.head_dim = 16
    cfg.vocab_size = len(tk)
    cfg.tie_word_embeddings = True
    model = AutoModelForCausalLM.from_config(cfg)

    args = SFTConfig(
        max_length=a.maxlen,
        per_device_train_batch_size=2,
        completion_only_loss=True,     # the thing under test
        packing=False,
        output_dir="/tmp/check_loss_mask",
        report_to="none",
        dataset_num_proc=1,
        bf16=False,
    )
    tr = SFTTrainer(model=model, train_dataset=ds, processing_class=tok, args=args)
    if (a.index_tokens or a.action_vocab or a.card_first) and not a.keep_eos:
        from sft_teacher import strip_trailing_eos
        tr.train_dataset = strip_trailing_eos(tr.train_dataset, tk.eos_token_id, "train")
    batch = next(iter(tr.get_train_dataloader()))
    ids, lab = batch["input_ids"], batch["labels"]
    print("[batch] keys %s | input_ids %s" % (sorted(batch.keys()), tuple(ids.shape)))

    bad = 0
    for r in range(ids.shape[0]):
        keep = (lab[r] != -100).nonzero().flatten().tolist()
        n = int((ids[r] != tk.pad_token_id).sum()) if tk.pad_token_id is not None else ids.shape[1]
        print("\n--- row %d | %d tokens, %d supervised ---" % (r, n, len(keep)))
        print("  supervised positions : %s" % keep)
        print("  supervised text      : %r" % tk.decode([ids[r][i] for i in keep]))
        print("  labels there         : %r" % tk.decode([lab[r][i] for i in keep]))
        # the tail of the prompt is the menu -- the single most important thing NOT to train on,
        # since reproducing the menu is trivial and would swamp the one token that matters
        first = keep[0] if keep else n
        print("  last 12 prompt tokens (must be unsupervised): %r"
              % tk.decode(ids[r][max(0, first - 12):first]))
        want = (None if a.card_first else
                (1 if ((a.index_tokens or a.action_vocab) and not a.keep_eos) else None))
        if not keep:
            bad += 1
            print("  *** NOTHING is supervised in this row ***")
        elif want is not None and len(keep) != want:
            bad += 1
            print("  *** %d supervised tokens, expected exactly %d (the answer alone) ***"
                  % (len(keep), want))
        elif a.card_first and not (1 <= len(keep) <= 2):
            bad += 1
            print("  *** %d supervised tokens; card-first answers are 1 or 2 ***"
                  % len(keep))
        elif want is None and not a.card_first and len(keep) > 4:
            bad += 1
            print("  *** %d supervised tokens -- the answer is ONE digit, so the prompt is "
                  "being trained on ***" % len(keep))

    print("\n[verdict] %s"
          % ("PASS: loss is on the completion only"
             if not bad else "FAIL: %d/%d rows are supervised outside the answer" % (bad, ids.shape[0])))
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
