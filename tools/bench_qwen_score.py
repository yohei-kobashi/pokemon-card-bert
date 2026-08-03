#!/usr/bin/env python3
"""Where the 0.199 s per decision goes, and what actually removes it.

instance2's 63-deck screen costs 5.5 hours. The scorer's own counter says 62,002 decisions at
0.199 s each = 3.4 h of that, so scoring is ~62% of the wall clock and the rest is the engine
opponent and the game loop.

0.199 s is roughly 5x what a ~800-token prefill of a 4B model should cost on this card, so the
first question is not "which serving framework" but "what is the time being spent on". Three
candidates, measured here rather than argued:

  merge      the LoRA is applied live (PeftModel), so every linear runs base + B(A(x)). Merging
             folds it into the base weights.
  kv         the card-first scheme needs a SECOND next-token distribution for candidate groups
             that share a first token. _score_card_first gets it by re-running the model on
             prompt + 1 token -- a full ~800-token prefill to score one extra position. The KV
             cache from the first forward makes it a single decode step.
  batch      decisions are scored one at a time. The screen plays games sequentially, so the
             card sees one ~800-token sequence at a time and is mostly idle.

Run with nothing else on the GPU; the earlier bench2 sweep was invalidated by a generation job
holding the CPU (see [[vast-cpu-quotas]]).
"""
import argparse
import gzip
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

_OPT = re.compile(r"(?:^| )(\d+)=(\S+)")


def load_decisions(path, n, minc=2):
    """Real (prompt, candidates) pairs out of a built SFT pool."""
    out = []
    with gzip.open(path, "rt") as f:
        for line in f:
            d = json.loads(line)
            pr = d["prompt"]
            cands = [t for _, t in _OPT.findall(pr.rsplit(":: ", 1)[-1])]
            if len(cands) < minc:
                continue
            out.append((pr, cands))
            if len(out) >= n:
                break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--maxlen", type=int, default=1024)
    ap.add_argument("--batches", default="1,4,8,16,32")
    a = ap.parse_args()

    import torch
    from tools.mirror_match import QwenScorer
    from lm.action_token import first_token, second_token, equivalent

    dec = load_decisions(a.data, a.n)
    print("[data] %d decisions | mean %.0f candidates" % (
        len(dec), sum(len(c) for _, c in dec) / len(dec)), flush=True)

    sc = QwenScorer(a.ckpt, maxlen=a.maxlen)
    tk, torch_ = sc.tk, sc.torch
    lens = [len(tk(p, add_special_tokens=False)["input_ids"]) for p, _ in dec]
    lens.sort()
    print("[data] prompt tokens: mean %.0f  p50 %d  p90 %d  max %d"
          % (sum(lens) / len(lens), lens[len(lens) // 2], lens[int(.9 * len(lens))], lens[-1]),
          flush=True)

    def timeit(fn, label, warm=3):
        for i in range(warm):
            fn(dec[i])
        torch.cuda.synchronize()
        t0 = time.time()
        for d in dec:
            fn(d)
        torch.cuda.synchronize()
        el = time.time() - t0
        print("  %-28s %7.1f ms/decision   (%.1fx)" % (label, 1000 * el / len(dec),
                                                       BASE[0] / (el / len(dec))
                                                       if BASE[0] else 1.0), flush=True)
        return el / len(dec)

    BASE = [0.0]

    # ---- A: exactly what the screen runs today -------------------------------------------
    print("\n[A] current path (PeftModel, second token re-prefills)", flush=True)
    BASE[0] = timeit(lambda d: sc._score_card_first(*d), "baseline")
    base = BASE[0]
    BASE[0] = base

    def report(el, label):
        print("  %-28s %7.1f ms/decision   %.2fx" % (label, 1000 * el, base / el), flush=True)

    # ---- B: fold the LoRA into the base weights ------------------------------------------
    print("\n[B] + merged LoRA", flush=True)
    try:
        merged = sc.model.merge_and_unload()
        merged.eval()
        sc.model = merged
        el = 0.0
        for i in range(3):
            sc._score_card_first(*dec[i])
        torch.cuda.synchronize(); t0 = time.time()
        for d in dec:
            sc._score_card_first(*d)
        torch.cuda.synchronize(); el = (time.time() - t0) / len(dec)
        report(el, "merged")
    except Exception as e:
        print("  merge FAILED: %s" % e, flush=True)
        el = base

    # ---- C: reuse the prompt's KV cache for the tie-break token ---------------------------
    model = sc.model
    dev = model.device

    def score_kv(d, use_cache=True):
        prompt, cands = d
        from lm.action_token import to_scheme_b
        p = to_scheme_b(prompt) if sc.scheme_b else prompt
        heads = [first_token(c) for c in cands]
        ids = tk(p, add_special_tokens=False, truncation=True, max_length=a.maxlen)["input_ids"]
        t = torch.tensor([ids], device=dev)
        with torch.no_grad():
            out = model(input_ids=t, use_cache=use_cache)
            lp1 = torch.log_softmax(out.logits[0, -1, :].float(), -1)
        hid = {h: tk.convert_tokens_to_ids(h) for h in set(heads)}
        sc_out = [float(lp1[hid[h]]) if hid[h] is not None else -1e9 for h in heads]
        need = {}
        for h in set(heads):
            grp = [i for i, x in enumerate(heads) if x == h]
            if len(grp) > 1 and not all(equivalent(cands[i], cands[grp[0]]) for i in grp):
                need[h] = grp
        for h, grp in need.items():
            with torch.no_grad():
                if use_cache:
                    import copy
                    pkv = copy.copy(out.past_key_values)
                    o2 = model(input_ids=torch.tensor([[hid[h]]], device=dev),
                               past_key_values=pkv, use_cache=False)
                else:
                    o2 = model(input_ids=torch.tensor([ids + [hid[h]]], device=dev))
                lp2 = torch.log_softmax(o2.logits[0, -1, :].float(), -1)
            for i in grp:
                t2 = second_token(cands[i]) if sc.scheme_b else None
                j = tk.convert_tokens_to_ids(t2) if t2 else None
                sc_out[i] += float(lp2[j]) if j is not None else -1e9
        return sc_out

    print("\n[C] + KV cache for the tie-break token", flush=True)
    ok = True
    try:
        ref = sc._score_card_first(*dec[0])
        got = score_kv(dec[0])
        d0 = max(abs(x - y) for x, y in zip(ref, got))
        print("  max |diff| vs baseline on one decision: %.5f" % d0, flush=True)
        ok = d0 < 1e-2
        if not ok:
            print("  !! KV path does NOT reproduce the baseline -- not usable", flush=True)
    except Exception as e:
        print("  KV path FAILED: %s" % e, flush=True)
        ok = False
    if ok:
        for i in range(3):
            score_kv(dec[i])
        torch.cuda.synchronize(); t0 = time.time()
        for d in dec:
            score_kv(d)
        torch.cuda.synchronize()
        report((time.time() - t0) / len(dec), "merged + kv")

    # ---- D: batch whole decisions ---------------------------------------------------------
    print("\n[D] + batching the first forward across decisions", flush=True)
    pad = tk.pad_token_id if tk.pad_token_id is not None else 0
    enc = [tk(p, add_special_tokens=False, truncation=True,
              max_length=a.maxlen)["input_ids"] for p, _ in dec]

    def batch_first(idx):
        """The first forward only -- the part every decision needs."""
        chunk = [enc[i] for i in idx]
        L = max(len(x) for x in chunk)
        t = torch.full((len(chunk), L), pad, dtype=torch.long, device=dev)
        m = torch.zeros((len(chunk), L), dtype=torch.long, device=dev)
        for r, x in enumerate(chunk):          # LEFT pad so -1 is the real last token
            t[r, L - len(x):] = torch.tensor(x, device=dev)
            m[r, L - len(x):] = 1
        with torch.no_grad():
            o = model(input_ids=t, attention_mask=m, use_cache=False)
            return torch.log_softmax(o.logits[:, -1, :].float(), -1)

    for B in [int(x) for x in a.batches.split(",") if x]:
        try:
            for _ in range(2):
                batch_first(list(range(min(B, len(dec)))))
            torch.cuda.synchronize(); t0 = time.time()
            n = 0
            for s in range(0, len(dec), B):
                idx = list(range(s, min(s + B, len(dec))))
                batch_first(idx); n += len(idx)
            torch.cuda.synchronize()
            el = (time.time() - t0) / n
            print("  batch %-3d first-forward only   %7.1f ms/decision   %.2fx"
                  % (B, 1000 * el, base / el), flush=True)
        except RuntimeError as e:
            print("  batch %-3d FAILED: %s" % (B, str(e)[:90]), flush=True)
            break
    print("\npeak VRAM %.1f GiB" % (torch.cuda.max_memory_allocated() / 2**30), flush=True)


if __name__ == "__main__":
    main()
