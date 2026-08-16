#!/usr/bin/env python3
"""Train the reranker toward the game plan: put the probability mass on conforming candidates.

    loss = -w * log( sum_{i in conformant} softmax(scores)_i )  +  l2sp * ||theta - theta_0||^2

The set form matters: several menu entries can satisfy one rule (two benched Dreepy to choose
between, three legal energy targets on the line), and singling one out would train against
answers the plan calls equally correct.

L2-SP is on by default here. This is a full fine-tune, and every full fine-tune of this model
so far has walked out of its basin -- the interpolation between d41_r8 and its first
continuation DIPPED (36.7-40.4% against endpoints of 45.2 and 41.5), which is what "no longer
linearly connected" looks like. Anchoring to the round's own init is the cheap half of EWC and
is the axis rehearsal cannot reach.

    PYTHONPATH=cg-lib python3 tools/dusk_plan_train.py --data /root/rl/plan_r1.jsonl.gz \\
        --model /root/out/d41_r8 --out /root/out/plan_r1 --probe
"""

import argparse
import gzip
import json
import math
import os
import random
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--maxlen", type=int, default=512)
    ap.add_argument("--l2sp", type=float, default=1e-3)
    ap.add_argument("--eval-frac", type=float, default=0.05)
    ap.add_argument("--probe", action="store_true",
                    help="overfit 300 rows, no anchor: a trainer that cannot collapse this "
                         "must not spend a round")
    a = ap.parse_args()

    rows = [json.loads(x) for x in gzip.open(a.data, "rt")]
    if not rows:
        sys.exit("no rows")
    random.Random(0).shuffle(rows)
    if a.probe:
        # Only force what the probe MEANS: a small set, no anchor. Overriding lr/epochs too
        # silently discarded every command-line value, so a three-arm sweep ran the same
        # configuration three times and reported its own re-run noise as a result. Defaults
        # are applied only where the caller did not ask for something.
        rows, a.l2sp = rows[:300], 0.0
        if a.lr == ap.get_default("lr"):
            a.lr = 3e-5
        if a.epochs == ap.get_default("epochs"):
            a.epochs = 8.0
        print("[probe] 300 rows | lr %.1e | epochs %.0f | accum %d | floor = target entropy"
              % (a.lr, a.epochs, a.accum), flush=True)
    n_ev = 0 if a.probe else max(50, int(len(rows) * a.eval_frac))
    ev, tr = rows[:n_ev], rows[n_ev:]
    print("[data] %d train / %d held-out" % (len(tr), len(ev)), flush=True)

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.truncation_side = "left"          # the menu is last; a right cut deletes the options
    # fp32 WEIGHTS, bf16 MATH. transformers loads a checkpoint in the dtype it was saved in, and
    # these are saved bf16, where a weight near 0.02 has a ulp of ~1.5e-4 -- larger than the
    # AdamW step this trainer takes, so the update rounds to zero and the model does not move.
    # Measured: eight rows of a single rule, 40 passes, lr 1e-5 -- bf16 wanders between 6/8 and
    # 1/8 correct forever, fp32 reaches 8/8 on pass five. bf16 also cannot REPRESENT the answer:
    # the real candidates of a decision differ by ~0.004 in logit and the bf16 grid near 1.0 is
    # 0.0078 wide, so the ranking is quantised away before the loss ever sees it.
    model = AutoModelForSequenceClassification.from_pretrained(
        a.model, dtype=torch.float32).to(dev)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    ref = None
    if a.l2sp > 0:
        ref = {n: p.detach().clone().float() for n, p in model.named_parameters()
               if p.requires_grad}
        print("[l2sp] anchored %d tensors at %.2g" % (len(ref), a.l2sp), flush=True)

    def scores(r):
        enc = tok([r["prompt"]] * len(r["cands"]), r["cands"], return_tensors="pt",
                  padding=True, truncation=True, max_length=a.maxlen).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            logits = model(**enc).logits.squeeze(-1)
        return logits.float()

    def acc(rs):
        model.eval()
        hit = 0
        with torch.no_grad():
            for r in rs:
                s = scores(r)
                best = max(range(len(r["wc"])), key=lambda j: r["wc"][j])
                hit += 1 if r["wc"][int(s.argmax())] >= r["wc"][best] else 0
        model.train()
        return hit / max(1, len(rs))

    if ev:
        print("[eval] plan-conformance before: %.1f%%" % (100 * acc(ev)), flush=True)
    elif a.probe:
        # The probe holds nothing out on purpose -- it asks whether the rule is REPRESENTABLE,
        # and that question is answered by whether the trainer can memorise it, not by whether
        # it generalises. Without this the only readout was the loss, which falls whenever the
        # target entropy is low regardless of where the argmax lands.
        print("[probe] train-set conformance before: %.1f%%" % (100 * acc(tr[:200])), flush=True)

    losses, step, n_oom = [], 0, 0
    n_steps = int(len(tr) * a.epochs)
    for i in range(n_steps):
        r = tr[i % len(tr)]
        if len(r["cands"]) < 2 or sum(r.get("wc") or []) <= 0:
            continue
        try:
            s = scores(r)
        except torch.OutOfMemoryError:
            # One row is one batch of len(cands) full-length pairs, and the tail runs to 24
            # candidates. Dropping the row costs one gradient; letting it kill the run costs
            # the whole training.
            n_oom += 1
            opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
            continue
        lp = torch.log_softmax(s, dim=-1)
        # Soft target: the merged plan weight over candidates, normalised. Rules that agree on
        # a candidate stack; rules that disagree split the target instead of contradicting.
        tgt = torch.tensor(r["wc"], device=dev, dtype=lp.dtype)
        tgt = tgt / tgt.sum()
        mass = (tgt * lp).sum()
        loss = -mass
        (loss / a.accum).backward()
        losses.append(float(-mass))
        step += 1
        if step % a.accum == 0:
            if ref is not None:
                # L2-SP applied as a GRADIENT, not as a term in the loss. Building
                #     pen = sum_n ((p_n - ref_n) ** 2).sum()
                # inside the autograd graph materialises a full-size intermediate for each of
                # the 202 anchored tensors and keeps them all alive until backward; that fits
                # in bf16 and OOM'd a 23.52 GiB card the moment fp32 doubled every activation.
                # d/dtheta of lambda*||theta - theta_0||^2 is 2*lambda*(theta - theta_0), which
                # goes straight onto .grad with no graph -- the same update in constant memory.
                # Added once per OPTIMISER step, not per micro-step: the data gradient is scaled
                # by 1/accum on the way in, so the anchor must not be scaled with it.
                with torch.no_grad():
                    for n_, p_ in model.named_parameters():
                        if p_.grad is not None and n_ in ref:
                            p_.grad.add_(p_.detach() - ref[n_], alpha=2.0 * a.l2sp)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad()
        if step % 500 == 0:
            pk = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
            print("  step %d  -log P(plan) %.4f  peak %.2f GiB%s"
                  % (step, st.mean(losses[-500:]), pk,
                     "  OOM-skipped %d" % n_oom if n_oom else ""), flush=True)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

    first, last = st.mean(losses[:200] or [0]), st.mean(losses[-200:] or [0])
    print("FINAL -log P(plan) %.4f -> %.4f over %d steps" % (first, last, step))
    if a.probe:
        after = 100 * acc(tr[:200])
        # The floor is the target's own entropy, not zero: a row whose plan weight is spread
        # over three candidates cannot be driven below that however well it is fitted. Printing
        # it next to the loss is what stops "1.09" from being read as failure when it is the
        # best the row admits.
        floors = []
        for r in tr[:200]:
            w = [x for x in r["wc"]]
            s_ = sum(w)
            if s_ > 0:
                p = [x / s_ for x in w if x > 0]
                floors.append(-sum(x * math.log(x) for x in p))
        fl = st.mean(floors) if floors else 0.0
        print("[probe] train-set conformance after:  %.1f%%" % after, flush=True)
        # Pass on the fraction of the AVAILABLE gap closed, not on an absolute drop. The old
        # test wanted 0.10 of loss; with soft targets there is often less than that in the whole
        # problem -- measured on a real round, chance is 0.6931 and the floor 0.6596, so the
        # entire range is 0.0335 and the probe could only ever print FAILED. A safety check that
        # cannot pass never told us anything, including during the nine rounds it watched the
        # loop train at exactly chance.
        room = max(1e-6, first - fl)
        print("PROBE %s | loss %.4f -> %.4f | floor %.4f | gap %.4f (%.0f%% of the %.4f "
              "available) | conformance %.1f%%"
              % ("OK" if (first - last) >= 0.25 * room else "FAILED",
                 first, last, fl, last - fl, 100 * (first - last) / room, room, after))
        return
    if ev:
        print("[eval] plan-conformance after:  %.1f%%" % (100 * acc(ev)), flush=True)
    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print("[saved] %s" % a.out)


if __name__ == "__main__":
    main()
