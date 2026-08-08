#!/usr/bin/env python3
"""DPO for the 4B on playout-measured preference pairs. 100% RL data, no SFT mixing.

WHY THIS EXISTS. Every training round to date consumed the playout measurements as SFT: take
argmax-Q, cross-entropy on that index, dilute to 2-10% in a 90%+ base mix. That threw away the
Q magnitudes, and the base majority was re-imitation of engine_v2 -- the recipe whose extra
epoch measurably made the policy WORSE (Stage-1 r1: -5.59pt +- 2.06, 10 of 11 decks down).
This trainer is the standard post-SFT move instead: preference pairs, reference-anchored.

    pair    (chosen, rejected) from tools/dpo_branch.py -- the playouts' verdict at a decision
            the policy itself was unsure about, gated by attach_label's four tests
    loss    -log sigma(beta * [(pi-ref)(y_w|x) - (pi-ref)(y_l|x)])   (+ optional cDPO smoothing
            for label noise: the Q gap is measured with SE ~0.1-0.25 at 16 playouts)
    ref     THE STARTING CHECKPOINT ITSELF. LoRA + embeddings are restored from --init-from
            before any step, so at step 0 policy == reference exactly; reference logprobs are
            precomputed then and cached. No second model in memory, no approximation drift.
            The KL anchor that base-mixing was imitating badly is exactly beta.

MODEL SETUP IS IMPORTED FROM sft_teacher, NOT COPIED: add_domain_tokens / warm_start /
unfreeze_new_rows / report_embedding_movement and the scheme-B label machinery are the same
functions the SFT rounds ran, so a checkpoint saved here warm-starts the next round through
the identical path (a copy drifting from the original is how silent label bugs survive review
-- the same reason qlabel_gen imports attach_label.label instead of reimplementing it).

    python3 tools/instance/dpo_teacher.py --data /root/dpo_r1.jsonl.gz \\
        --init-from /root/out/i2_r7 --card-first /root/ptcg/repo/data/cardfirst_b_v39.json \\
        --domain-tokens --out /root/out/dpo_r1
    # sanity first: --probe trains on 2k pairs until it overfits; a trainer that cannot drive
    # THAT loss to ~0 has an optimisation problem and must not spend a real round
"""

import argparse
import gzip
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, os.path.join(ROOT, "cg-lib"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sft_teacher import (TARGETS, add_domain_tokens, option_texts,   # noqa: E402
                         unfreeze_new_rows, warm_start, report_embedding_movement)


def load_dpo_pairs(path, limit=0, cf_set=None):
    """dpo_branch rows -> (scheme-B prompt, completion_w, completion_l) triples.

    The conversion is the SAME path load_pairs takes for SFT (option_texts -> to_scheme_b ->
    label_b), run twice per row. Rows whose two menu indices collapse to the SAME answer token
    are dropped -- a preference between identical strings is unlearnable -- and counted, since
    a large number would mean dpo_branch and the dedup disagree about what a distinct act is.
    """
    from lm.action_token import to_scheme_b, label_b, first_token
    rows, drop = [], {"no_menu": 0, "bad_idx": 0, "same_completion": 0, "oov": 0}
    with gzip.open(path, "rt") as f:
        for line in f:
            d = json.loads(line)
            opts = option_texts(d["prompt"])
            if opts is None:
                drop["no_menu"] += 1
                continue
            iw, il = int(d["tw"]), int(d["tl"])
            if not (0 <= iw < len(opts) and 0 <= il < len(opts)):
                drop["bad_idx"] += 1
                continue
            if cf_set is not None and (first_token(opts[iw]) not in cf_set
                                       or first_token(opts[il]) not in cf_set):
                drop["oov"] += 1
                continue
            aw, bw = label_b(d["prompt"], iw, opts)
            al, bl = label_b(d["prompt"], il, opts)
            cw, cl = aw + (bw or ""), al + (bl or "")
            if cw == cl:
                drop["same_completion"] += 1
                continue
            rows.append({"prompt": to_scheme_b(d["prompt"]), "cw": cw, "cl": cl,
                         "q_gap": float(d.get("qw", 0)) - float(d.get("ql", 0)),
                         "pl": int(d.get("pl", 0) or 0),
                         "model_was": d.get("model_was", "?"), "deck": d.get("deck")})
            if limit and len(rows) >= limit:
                break
    print("[data] %d pairs | dropped %s" % (len(rows), drop), flush=True)
    return rows


def completion_logprobs(model, tok, torch, batch, maxlen, dev="cuda"):
    """Sum log p(completion tokens | prompt) for each (prompt, completion) in `batch`."""
    tk = getattr(tok, "tokenizer", tok)
    seqs, spans = [], []
    for prompt, comp in batch:
        p_ids = tk(prompt, add_special_tokens=True)["input_ids"]
        c_ids = tk(comp, add_special_tokens=False)["input_ids"]
        ids = (p_ids + c_ids)[-maxlen:]
        nc = min(len(c_ids), len(ids) - 1)   # completion tokens that survived the left-trim
        seqs.append(ids)
        spans.append(nc)
    L = max(len(s) for s in seqs)
    pad = tk.pad_token_id if tk.pad_token_id is not None else tk.eos_token_id
    inp = torch.full((len(seqs), L), pad, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):                      # LEFT padding: completions end at -1
        inp[i, L - len(s):] = torch.tensor(s)
        att[i, L - len(s):] = 1
    inp, att = inp.to(dev), att.to(dev)
    # Completions are 1-3 tokens, so only the last few positions matter -- slice BEFORE the
    # fp32 log_softmax. Materialising the whole [B, L, 154k] in fp32 is ~4 GB per forward and
    # is exactly the logits-bandwidth wall the 4B SFT already measured ([[v31-sft-vast-run]]).
    # Left padding + RoPE makes the uniform position shift exact, so the slice is safe.
    maxc = max(1, max(spans))
    logits = model(input_ids=inp, attention_mask=att).logits[:, -(maxc + 1):-1].float()
    lsm = torch.log_softmax(logits, dim=-1)                          # [B, maxc, V]
    tgt = inp[:, -maxc:]
    tok_lp = lsm.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)          # [B, maxc]
    out = []
    for i, nc in enumerate(spans):
        out.append(tok_lp[i, -nc:].sum() if nc > 0 else tok_lp[i].sum() * 0)
    return torch.stack(out)                                          # [B]


# Agreement between two INDEPENDENT playout measurements of the same branch point, measured
# 2026-08-08 by re-running dpo_branch on round 2's exact 20,000 points with only --seed
# changed. u is the gap in 16-playout units (1 unit = one playout flipping), and the u=2/u=3
# cells are pooled because they violated monotonicity (PAVA on n=207/371).
_AGREE = [(3, 0.704), (4, 0.796), (5, 0.843), (6, 0.904), (7, 0.934), (8, 0.993)]


def eps_for_gap(q_gap, playouts):
    """Per-pair cDPO epsilon = P(this label is the wrong way round).

    Two steps, and the second is the one that is easy to get wrong. (1) Put the gap on a
    common scale: the standard error of a playout mean goes as 1/sqrt(P), so a gap measured
    with P playouts is worth sqrt(P/16) units on the 16-playout curve -- without this a
    64-playout label would be charged the noise of a 16-playout one. (2) AGREEMENT IS NOT
    ACCURACY. If each measurement is right with probability a, two of them agree with
    probability a^2 + (1-a)^2, so 82.7% agreement means a = 90.4%, not 82.7%. Feeding raw
    agreement in as epsilon would roughly double the smoothing and flatten real labels.
    """
    u = abs(q_gap) * 8.0 * ((max(1, playouts) / 16.0) ** 0.5)
    if u <= _AGREE[0][0]:
        agree = _AGREE[0][1]
    elif u >= _AGREE[-1][0]:
        agree = _AGREE[-1][1]
    else:
        for (u0, a0), (u1, a1) in zip(_AGREE, _AGREE[1:]):
            if u0 <= u <= u1:
                agree = a0 + (a1 - a0) * (u - u0) / (u1 - u0)
                break
    a = 0.5 * (1.0 + max(0.0, 2.0 * agree - 1.0) ** 0.5)
    return min(0.45, max(0.005, 1.0 - a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen3-4B-Base")
    ap.add_argument("--data", required=True, help="dpo_branch.py output")
    ap.add_argument("--out", required=True)
    ap.add_argument("--init-from", required=True, help="the checkpoint the POLICY starts from")
    ap.add_argument("--ref-from", default="",
                    help="checkpoint the REFERENCE policy comes from. Default: --init-from, "
                         "which is the cheap case (policy == ref at step 0, so no second model "
                         "is needed). Point it at the SFT checkpoint to keep beta's KL leash "
                         "anchored THERE across rounds. This matters because SFT and DPO share "
                         "one rank-16 adapter -- warm_start restores it and DPO trains the same "
                         "tensors -- so LoRA bounds drift from the BASE model, NOT from the SFT "
                         "result. Re-anchoring the reference each round removes the only thing "
                         "that was holding the SFT in place.")
    ap.add_argument("--card-first", required=True)
    ap.add_argument("--domain-tokens", action="store_true", required=False, default=True)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--cdpo-eps", type=float, default=0.0,
                    help="label-noise smoothing: eps fraction assumed mislabelled")
    ap.add_argument("--cdpo-calibrated", action="store_true",
                    help="per-pair epsilon from the pair's own playout gap (see eps_for_gap). "
                         "Overrides --cdpo-eps. This is the answer to the measurement that "
                         "only ~35%% of round 2's pairs reproduced with the same verdict.")
    ap.add_argument("--label-playouts", type=int, default=16,
                    help="playouts behind qw/ql when a row predates the 'pl' field")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bsz", type=int, default=8, help="PAIRS per forward (2x sequences)")
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--maxlen", type=int, default=896)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--eval-frac", type=float, default=0.05)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--probe", action="store_true",
                    help="overfit sanity: cap at 2000 pairs, 30 epochs, expect loss -> ~0")
    ap.add_argument("--seed", type=int, default=3407)
    a = ap.parse_args()
    if a.probe:
        # 1000 pairs x 10 epochs ~ 40 min: enough passes over the same data that a healthy
        # optimiser visibly collapses the loss, small enough to be a sanity check and not a run
        a.limit, a.epochs = min(a.limit or 1000, 1000), 10.0

    t0 = time.time()
    from unsloth import FastLanguageModel               # noqa: E402  (before transformers)
    import torch                                        # noqa: E402
    import random

    print("[stack] torch %s cuda=%s | model %s" % (torch.__version__,
                                                   torch.cuda.is_available(), a.model), flush=True)
    model, tok = FastLanguageModel.from_pretrained(
        model_name=a.model, max_seq_length=a.maxlen,
        load_in_4bit=False, load_in_16bit=True, full_finetuning=False)

    _cf = json.load(open(a.card_first))
    cf_set = set(_cf["first_tokens"])
    if _cf.get("scheme") != "b":
        raise SystemExit("this trainer assumes card-first scheme B (what i2_r7 was trained on)")
    cf_extra = list(_cf["new_tokens"]) + list(_cf["second_tokens"])
    n_base_vocab = add_domain_tokens(model, tok, torch, False, None, cf_extra)

    model = FastLanguageModel.get_peft_model(
        model, r=a.rank, target_modules=TARGETS, lora_alpha=a.alpha, lora_dropout=0,
        bias="none", use_gradient_checkpointing="unsloth", random_state=a.seed,
        max_seq_length=a.maxlen)
    warm_start(model, getattr(tok, "tokenizer", tok), torch, a.init_from, n_base_vocab or 0)
    emb0, base0 = unfreeze_new_rows(model, n_base_vocab, torch, True)
    print("[peft] trainable %.1fM"
          % (sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6), flush=True)

    rows = load_dpo_pairs(a.data, a.limit, cf_set)
    if len(rows) < 50:
        raise SystemExit("too few pairs (%d) -- not spending a round on this" % len(rows))
    if a.cdpo_calibrated:
        import statistics as _st
        for r in rows:
            r["eps"] = eps_for_gap(r["q_gap"], r["pl"] or a.label_playouts)
        e = sorted(r["eps"] for r in rows)
        print("[cdpo] per-pair eps: mean %.3f p10 %.3f p50 %.3f p90 %.3f | pairs at the 0.45 "
              "cap %d (their labels are coin flips and are held near-neutral)"
              % (_st.mean(e), e[len(e) // 10], e[len(e) // 2], e[9 * len(e) // 10],
                 sum(1 for x in e if x >= 0.4499)), flush=True)
    random.Random(a.seed).shuffle(rows)
    n_ev = max(20, int(len(rows) * a.eval_frac)) if not a.probe else 0
    ev, tr_rows = rows[:n_ev], rows[n_ev:]
    was = {"w": 0, "l": 0}
    for r in tr_rows:
        was[r["model_was"]] = was.get(r["model_was"], 0) + 1
    print("[data] train %d (model right %d / wrong %d) | held-out %d"
          % (len(tr_rows), was.get("w", 0), was.get("l", 0), n_ev), flush=True)

    # ---- reference logprobs -------------------------------------------------------------
    # Default: the policy IS the reference at step 0, so one forward pass over the loaded
    # weights is exactly log pi_ref and no second model is needed. With --ref-from, warm-start
    # to the REFERENCE first, take the same pass, then warm-start back to the policy init --
    # still one model in memory, at the cost of one extra adapter load.
    if a.ref_from and a.ref_from != a.init_from:
        print("[ref] anchoring beta at %s (policy starts from %s)"
              % (a.ref_from, a.init_from), flush=True)
        warm_start(model, getattr(tok, "tokenizer", tok), torch, a.ref_from, n_base_vocab or 0)
    model.eval()
    ref = {}
    with torch.no_grad():
        for i in range(0, len(rows), a.bsz):
            chunk = rows[i:i + a.bsz]
            lw = completion_logprobs(model, tok, torch,
                                     [(r["prompt"], r["cw"]) for r in chunk], a.maxlen)
            ll = completion_logprobs(model, tok, torch,
                                     [(r["prompt"], r["cl"]) for r in chunk], a.maxlen)
            for j, r in enumerate(chunk):
                ref[id(r)] = (float(lw[j]), float(ll[j]))
    print("[ref] cached %d pairs (+%.1fs)" % (len(ref), time.time() - t0), flush=True)
    if a.ref_from and a.ref_from != a.init_from:
        # Back to the policy's own starting weights. Without this the round would train the
        # REFERENCE checkpoint and report it as a continuation of --init-from.
        warm_start(model, getattr(tok, "tokenizer", tok), torch, a.init_from, n_base_vocab or 0)
        print("[ref] policy restored to %s" % a.init_from, flush=True)

    def eval_pairs(rs):
        if not rs:
            return float("nan"), float("nan")
        model.eval()
        losses, acc = [], 0
        with torch.no_grad():
            for i in range(0, len(rs), a.bsz):
                chunk = rs[i:i + a.bsz]
                pw = completion_logprobs(model, tok, torch,
                                         [(r["prompt"], r["cw"]) for r in chunk], a.maxlen)
                pl = completion_logprobs(model, tok, torch,
                                         [(r["prompt"], r["cl"]) for r in chunk], a.maxlen)
                for j, r in enumerate(chunk):
                    rw, rl = ref[id(r)]
                    z = a.beta * ((float(pw[j]) - rw) - (float(pl[j]) - rl))
                    losses.append(-torch.nn.functional.logsigmoid(torch.tensor(z)).item())
                    acc += z > 0
        model.train()
        return sum(losses) / len(losses), 100.0 * acc / len(rs)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=a.lr, weight_decay=0.0)
    model.train()
    step = seen = 0
    losses, accs = [], []
    n_total = int(len(tr_rows) * a.epochs)
    t1 = time.time()
    order = list(range(len(tr_rows)))
    while seen < n_total:
        random.Random(a.seed + step).shuffle(order)
        for i in range(0, len(order), a.bsz):
            if seen >= n_total:
                break
            chunk = [tr_rows[k] for k in order[i:i + a.bsz]]
            pw = completion_logprobs(model, tok, torch,
                                     [(r["prompt"], r["cw"]) for r in chunk], a.maxlen)
            pl = completion_logprobs(model, tok, torch,
                                     [(r["prompt"], r["cl"]) for r in chunk], a.maxlen)
            rw = torch.tensor([ref[id(r)][0] for r in chunk], device=pw.device)
            rl = torch.tensor([ref[id(r)][1] for r in chunk], device=pw.device)
            z = a.beta * ((pw - rw) - (pl - rl))
            loss_pos = -torch.nn.functional.logsigmoid(z)
            if a.cdpo_calibrated:
                eps = torch.tensor([r["eps"] for r in chunk], device=z.device, dtype=z.dtype)
                loss = ((1 - eps) * loss_pos
                        - eps * torch.nn.functional.logsigmoid(-z)).mean()
            elif a.cdpo_eps > 0:
                loss = ((1 - a.cdpo_eps) * loss_pos
                        - a.cdpo_eps * torch.nn.functional.logsigmoid(-z)).mean()
            else:
                loss = loss_pos.mean()
            (loss / a.accum).backward()
            losses.append(float(loss))
            accs.append(float((z > 0).float().mean()))
            seen += len(chunk)
            step += 1
            if step % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad()
            if step % 50 == 0:
                print("  step %d seen %d/%d | loss %.4f | acc %.1f%% | %.0fs"
                      % (step, seen, n_total, sum(losses[-50:]) / min(50, len(losses)),
                         100 * sum(accs[-50:]) / min(50, len(accs)), time.time() - t1),
                      flush=True)

    vl, va = eval_pairs(ev)
    print("FINAL train loss %.4f acc %.1f%% | held-out loss %.4f acc %.1f%% | seen %d"
          % (sum(losses[-100:]) / min(100, len(losses)),
             100 * sum(accs[-100:]) / min(100, len(accs)), vl, va, seen), flush=True)
    if a.probe:
        final = sum(losses[-100:]) / min(100, len(losses))
        print("PROBE %s: loss %.4f (start ~0.69). A trainer that cannot overfit 2k pairs has "
              "an optimisation problem." % ("OK" if final < 0.15 else "FAILED", final),
              flush=True)
        return

    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    import shutil
    shutil.copyfile(a.card_first, os.path.join(a.out, "cardfirst_vocab.json"))
    if n_base_vocab is not None and emb0 is not None:
        report_embedding_movement(model, n_base_vocab, emb0, torch, a.out, base0)
    print("[peak] VRAM %.1f GiB" % (torch.cuda.max_memory_allocated() / 2**30), flush=True)
    print("[saved] %s | total %.1f min" % (a.out, (time.time() - t0) / 60), flush=True)
    print("DPO_TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
