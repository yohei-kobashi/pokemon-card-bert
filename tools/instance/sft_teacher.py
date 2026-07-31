"""SFT the Qwen3.5-9B-Base teacher on engine_v2 demonstrations, via Unsloth bf16 LoRA.

Follows Unsloth's published Qwen3.5 fine-tuning recipe (FastLanguageModel, load_in_16bit,
explicit target_modules, max_seq_length passed to get_peft_model too, adamw_8bit, warmup_steps,
dataset_num_proc=1) and deviates only where this task requires it -- each deviation is annotated.

NOT 4-bit: Unsloth's guide says "It is not recommended to do QLoRA (4-bit) training on the
Qwen3.5 models", and bf16 LoRA for 9B needs ~22 GB against this card's 48 GB.

STACK REQUIREMENT (learned the hard way). Qwen3.5's linear-attention layers only take the fast
path when ALL FOUR of these import (transformers/models/qwen3_5/modeling_qwen3_5.py:205):
`causal_conv1d_fn`, `causal_conv1d_update`, `chunk_gated_delta_rule`,
`fused_recurrent_gated_delta_rule` -- i.e. BOTH `causal-conv1d` AND `flash-linear-attention`.
fla's cpp extensions additionally need torch >= 2.11. On a CUDA-12.8 driver that means
torch==2.11.0+cu128 specifically (the default `pip install unsloth` pulls a cu13 build the driver
cannot run). Without the fast path the DeltaNet layers fall back to a pure-torch recurrence and a
single step did not finish in 3.5 minutes.

Data: tools/_legacy_decoder/build_sft.py --target-mode index, so the completion is the chosen
option's MENU INDEX ("0".."51"), 1-2 tokens instead of 7-10 for the action string. Verified on all
3,088,497 records that menu[index] == the action string and prompts are byte-identical to the
action-target build.

Prompt/EOS contract: the data is a RAW COMPLETION format, which is why -Base is used -- no chat
template is applied anywhere, and the `prompt` string is exactly what lm/serialize renders at
inference. Loss is completion-only.

Smoke:  --limit 2000 --steps 20
Real:   --limit 200000 --epochs 1
"""
import argparse
import gzip
import json
import os
import re
import time

os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "0")   # keep fused CE; 248k-vocab logits are 1.3 GB

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_pairs(path, limit, skip=0):
    """`skip` records are dropped first, so the held-out slice never overlaps training."""
    P, C = [], []
    with gzip.open(path, "rt") as f:
        for i, line in enumerate(f):
            if i < skip:
                continue
            d = json.loads(line)
            t = d.get("target")
            if not t:
                continue
            P.append(d["prompt"])
            C.append(t)
            if limit and len(P) >= limit:
                break
    return {"prompt": P, "completion": C}


def n_options(prompt):
    """How many numbered options the rendered menu offers."""
    menu = prompt.rsplit(":: ", 1)[-1]
    return len(re.findall(r"(?:^| )(\d+)=", menu))


def eval_top1(model, tok, torch, pairs, maxlen, bsz=16):
    """Top-1 on held-out decisions, scored the way inference will score.

    Restricted to decisions with <= 10 options (84% of them, measured), because there the answer
    is exactly ONE token and the whole decision is a single argmax over the digit tokens -- which
    is also the deployment contract (constrain to legal indices, so an illegal move is
    impossible). Decisions with more options need a 2-token answer and are excluded rather than
    scored with a wrong first-token-only rule.
    """
    model.eval()
    # Qwen3.5 is a VLM, so `tok` is a PROCESSOR: a positional call makes the first argument
    # `images`, and our prompts came back as "Incorrect image source ... Got 0". Use the
    # underlying text tokenizer.
    tk = getattr(tok, "tokenizer", tok)
    if tk.pad_token is None:
        tk.pad_token = tk.eos_token
    tk.padding_side = "left"          # last position must be the answer slot
    idx_tok = {}
    for k in range(10):
        ids = tk(str(k), add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            idx_tok[k] = ids[0]
    ok = tot = skipped = 0
    P, C = pairs["prompt"], pairs["completion"]
    with torch.no_grad():
        for i in range(0, len(P), bsz):
            bp, bc = P[i:i + bsz], C[i:i + bsz]
            keep = [j for j, p in enumerate(bp) if n_options(p) <= 10 and bc[j].isdigit()
                    and int(bc[j]) in idx_tok]
            skipped += len(bp) - len(keep)
            if not keep:
                continue
            texts = [bp[j] for j in keep]
            enc = tk(texts, return_tensors="pt", padding=True, truncation=True,
                     max_length=maxlen).to(model.device)
            out = model(**enc).logits[:, -1, :].float()
            for r, j in enumerate(keep):
                n = n_options(bp[j])
                cand = [(k, idx_tok[k]) for k in range(n) if k in idx_tok]
                best = max(cand, key=lambda kt: out[r, kt[1]].item())[0]
                ok += int(best == int(bc[j]))
                tot += 1
    model.train()
    return ok, tot, skipped


def add_domain_tokens(model, tok, torch):
    """Extend the tokenizer with the SAME 2,971 domain tokens the reranker uses, and report
    whether the new rows are actually distinct vectors.

    History: on the reranker, 3,087 added tokens ended up at pairwise cosine +0.998 -- every
    card was the same vector to three decimals, so no training could ever have used DECK[].
    That came from initialising new rows at the embedding MEAN. transformers now seeds them
    from a multivariate normal matching the old embeddings' mean and covariance, which should
    give distinct vectors -- but that is the exact thing that failed before, so it is measured
    here rather than assumed.
    """
    import sys
    for p in ("/root/ptcg/repo", "/root/ptcg/repo/cg-lib"):
        if p not in sys.path:
            sys.path.insert(0, p)
    from lm.vocab import special_tokens

    tk = getattr(tok, "tokenizer", tok)
    n_old = len(tk)
    n_added = tk.add_tokens(special_tokens())
    model.resize_token_embeddings(len(tk))
    print("[tokens] vocab %d -> %d (%d genuinely new)" % (n_old, len(tk), n_added), flush=True)
    if n_added == 0:
        return None

    w = model.get_input_embeddings().weight
    # Re-initialise the new rows: RANDOM DIRECTIONS at the base row-norm scale. Measured
    # (tools/instance/probe_init.py) against the base rows' own geometry, mean pairwise cosine:
    #     BASE +0.0117 | default +1.0000 | subtoken +0.8587 | subtoken-centred +0.8583
    #     random-centred -0.0000            <- the only one that matches BASE
    # transformers' default leaves every new row the same near-zero vector (row-norm 0.092 vs
    # 0.855), i.e. the +0.998 collapse that made the reranker unable to read DECK[].
    # Sub-token means do not fix it: every card token shares the pieces "c" and digits, so they
    # share a direction no centring removes -- the earlier "semantic re-init" reached only
    # +0.7096 for the same reason. Random loses the (unusable) semantics and buys a geometry the
    # rows can actually be trained apart from, which is the point of training them at all.
    with torch.no_grad():
        tgt = w[:n_old].float().norm(dim=1).median()
        gen = torch.Generator(device="cpu").manual_seed(3407)
        g = torch.randn(n_added, w.shape[1], generator=gen, dtype=torch.float32)
        g = g / g.norm(dim=1, keepdim=True) * tgt
        w[n_old:] = g.to(device=w.device, dtype=w.dtype)
    new = w[n_old:].detach().float()
    nrm = new / new.norm(dim=1, keepdim=True).clamp_min(1e-9)
    # sample 512 rows: the full 2971^2 gram is affordable but the sample is enough and keeps
    # this cheap if the token set grows
    idx = torch.arange(0, new.shape[0], max(1, new.shape[0] // 512))[:512]
    g = nrm[idx] @ nrm[idx].T
    off = g[~torch.eye(len(idx), dtype=torch.bool, device=g.device)]
    base = w[:n_old].detach().float()
    bn = base[::max(1, n_old // 512)][:512]
    bn = bn / bn.norm(dim=1, keepdim=True).clamp_min(1e-9)
    bg = bn @ bn.T
    boff = bg[~torch.eye(len(bn), dtype=torch.bool, device=bg.device)]
    print("[tokens] new-row pairwise cosine  mean %+.4f  max %+.4f   (base rows: mean %+.4f)"
          % (off.mean(), off.max(), boff.mean()), flush=True)
    # Threshold is set from the BASE rows measured in the same run, not a guessed constant.
    # The first version used 0.9 and would have waved through the +0.8177 the smoke produced.
    if off.mean() > boff.mean() + 0.15:
        raise SystemExit("[tokens] REFUSING: new rows sit at mean cosine %.4f against %.4f for "
                         "the base rows. That is the degeneracy that made the reranker unable "
                         "to read DECK[]; training from a collapsed geometry wastes the run."
                         % (off.mean(), boff.mean()))
    return n_old


def unfreeze_new_rows(model, n_base, torch):
    """Train ONLY the added embedding rows.

    The full input embedding is 248,320 x 4,096 = 1.02B parameters; training all of it would
    cost more than the LoRA it is meant to support, and the base rows do not need to move.
    A gradient hook zeroes every base row, so the optimizer sees gradient only where the new
    tokens are. `lm_head` is left frozen: tie_word_embeddings is False on this model, and under
    the constrained-output rule the model only ever emits digits, never a card token.
    """
    emb = model.get_input_embeddings()
    emb.weight.requires_grad_(True)

    def _mask_grad(grad):
        grad = grad.clone()
        grad[:n_base] = 0
        return grad

    emb.weight.register_hook(_mask_grad)
    head = model.get_output_embeddings()
    if head is not None:
        head.weight.requires_grad_(False)
    print("[emb] training rows %d.. (%d of %d) | lm_head frozen"
          % (n_base, emb.weight.shape[0] - n_base, emb.weight.shape[0]), flush=True)
    # keep a SAMPLE of base rows (the full 248k x 4096 copy would be 4 GB) to prove afterwards
    # that the hook actually held them
    bidx = torch.arange(0, n_base, max(1, n_base // 4096))[:4096].to(emb.weight.device)
    return (emb.weight[n_base:].detach().float().cpu().clone(),
            (bidx.cpu(), emb.weight[bidx].detach().float().cpu().clone()))


def report_embedding_movement(model, n_base, emb0, torch, out_dir, base0=None):
    """Did the new rows actually LEARN, and did the base rows stay put?

    On the reranker the added rows shifted 0.42% of their norm in a full epoch even with
    --emb-lr-mult 3, i.e. they stayed at their hand-written initialisation. If that repeats
    here, the tokens were added for nothing and the run should not be read as a test of them.
    """
    w = model.get_input_embeddings().weight.detach().float().cpu()
    new = w[n_base:]
    d = (new - emb0).norm(dim=1) / emb0.norm(dim=1).clamp_min(1e-9)
    moved = (d > 1e-6)
    # Rows for tokens that never appeared in the sampled data get no gradient, so the median over
    # ALL rows reads 0.000% and hides whether the ones that DID appear learned anything. Report
    # the share that moved, and the distribution among those.
    print("[emb] new rows that moved: %d/%d (%.1f%%)"
          % (moved.sum(), len(d), 100.0 * moved.sum() / len(d)), flush=True)
    if moved.any():
        dm = d[moved]
        print("[emb]   among them: median %.3f%%  p90 %.3f%%  max %.3f%%  "
              "(the reranker's added rows managed 0.42%% and stayed unlearned)"
              % (100 * dm.median(), 100 * dm.quantile(0.9), 100 * dm.max()), flush=True)
    # The other half of the check: the gradient hook is supposed to hold every BASE row fixed.
    # Nothing so far has verified that, and a leak would mean a 1B-parameter finetune nobody
    # asked for, quietly degrading the base model.
    if base0 is not None:
        bd = (w[base0[0]] - base0[1]).abs().max()
        print("[emb] base rows max abs change %.3e  %s"
              % (bd, "OK (frozen)" if bd < 1e-6 else "*** LEAK: the grad hook is not holding ***"),
              flush=True)
    path = os.path.join(out_dir, "domain_embeddings.pt")
    torch.save({"n_base": n_base, "rows": w[n_base:].clone()}, path)
    print("[emb] saved %d new rows -> %s" % (new.shape[0], path), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen3.5-9B-Base")
    ap.add_argument("--data", default="/root/ptcg/repo/data/sft/teacher_0730_index.jsonl.gz")
    ap.add_argument("--out", default="/root/out/teacher9b")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=0, help="max_steps; 0 = use --epochs")
    ap.add_argument("--epochs", type=float, default=1.0)
    # 1024 not the recipe's 2048: measured p99 739 / max 897 tokens on this data, so 2048 would
    # pad-waste half of every batch.
    ap.add_argument("--maxlen", type=int, default=1024)
    # The recipe uses bsz 1 x accum 4 to fit 27B. 9B on 48 GB has room, and our sequences are
    # short; raise it for throughput and keep the same effective batch shape.
    ap.add_argument("--bsz", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)          # recipe: r=16, alpha=16
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--domain-tokens", action="store_true",
                    help="add lm.vocab.special_tokens() and train ONLY those embedding rows")
    ap.add_argument("--eval-n", type=int, default=4000,
                    help="held-out records taken from the FRONT of the file; training skips them")
    a = ap.parse_args()

    t0 = time.time()
    from unsloth import FastLanguageModel               # noqa: E402  (must precede transformers)
    import torch                                        # noqa: E402
    from datasets import Dataset                        # noqa: E402
    from trl import SFTTrainer, SFTConfig               # noqa: E402
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M   # noqa: E402

    print("[stack] torch %s cuda=%s | qwen3_5 FAST PATH = %s"
          % (torch.__version__, torch.cuda.is_available(), M.is_fast_path_available), flush=True)
    if not M.is_fast_path_available:
        print("[stack] REFUSING: without the fast path the DeltaNet layers run a pure-torch "
              "recurrence and a single step takes minutes. Install causal-conv1d + "
              "flash-linear-attention on torch>=2.11 (cu128 build for a CUDA 12.8 driver).",
              flush=True)
        raise SystemExit(2)

    model, tok = FastLanguageModel.from_pretrained(
        model_name=a.model,
        max_seq_length=a.maxlen,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )
    print("[load] %.1fs %s" % (time.time() - t0, type(model).__name__), flush=True)

    n_base_vocab = None
    if a.domain_tokens:
        n_base_vocab = add_domain_tokens(model, tok, torch)

    model = FastLanguageModel.get_peft_model(
        model,
        r=a.rank,
        target_modules=TARGETS,
        lora_alpha=a.alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=a.maxlen,
    )
    emb0 = base0 = None
    if n_base_vocab is not None:
        emb0, base0 = unfreeze_new_rows(model, n_base_vocab, torch)
    print("[peft] trainable %.1fM"
          % (sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6), flush=True)

    ev = load_pairs(a.data, a.eval_n) if a.eval_n else None
    d = load_pairs(a.data, a.limit, skip=a.eval_n)
    ds = Dataset.from_dict(d)
    print("[data] train %d | held-out %d | completions %r"
          % (len(ds), len(ev["prompt"]) if ev else 0, d["completion"][:8]), flush=True)
    if ev:
        ok, tot, sk = eval_top1(model, tok, torch, ev, a.maxlen)
        print("[eval BEFORE] top1 %d/%d = %.2f%%  (skipped %d with >10 options)"
              % (ok, tot, 100.0 * ok / max(1, tot), sk), flush=True)

    cfg = SFTConfig(
        max_length=a.maxlen,
        per_device_train_batch_size=a.bsz,
        gradient_accumulation_steps=a.accum,
        warmup_steps=a.warmup,
        max_steps=a.steps if a.steps else -1,
        num_train_epochs=a.epochs,
        learning_rate=a.lr,
        logging_steps=1,
        output_dir=a.out,
        optim="adamw_8bit",
        seed=3407,
        dataset_num_proc=1,
        save_steps=a.save_steps,
        save_total_limit=2,
        lr_scheduler_type="cosine",
        bf16=True,
        completion_only_loss=True,   # never train on the prompt
        packing=False,               # one decision per sample; packing would blur boundaries
        # Length grouping is NOT available: trl 0.24's SFTConfig has no `group_by_length`. Its
        # modern replacement `padding_free=True` concatenates sequences and relies on
        # flash-attention varlen, and 3 of every 4 Qwen3.5 layers are linear attention carrying a
        # RECURRENT STATE across the sequence -- nothing guarantees that state resets at the
        # concatenation boundaries. So ~30% padding waste is accepted rather than trading
        # correctness for throughput. `packing` is off for the same reason.
        report_to="none",
    )
    tr = SFTTrainer(model=model, train_dataset=ds, processing_class=tok, args=cfg)
    print("[train] start (+%.1fs)" % (time.time() - t0), flush=True)
    r = tr.train()
    print("[done] %s" % r.metrics, flush=True)
    if ev:
        ok, tot, sk = eval_top1(model, tok, torch, ev, a.maxlen)
        print("[eval AFTER ] top1 %d/%d = %.2f%%   GATE: beat the reranker's 69.7%%"
              % (ok, tot, 100.0 * ok / max(1, tot)), flush=True)
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    if n_base_vocab is not None and emb0 is not None:
        report_embedding_movement(model, n_base_vocab, emb0, torch, a.out, base0)
    print("[saved] %s | total %.1f min" % (a.out, (time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
