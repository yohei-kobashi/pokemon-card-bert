#!/usr/bin/env python3
"""Round 3. Round 2 said logits_to_keep is worth 1.13x and is exact; the KV path threw an
exception with an empty message, so this run prints the traceback instead of swallowing it.

It also tests a lever the first two rounds missed. After the forward, the current code does

    lp1 = torch.log_softmax(out.logits[0, -1, :].float(), -1)
    base = [float(lp1[hid[h]]) for h in heads]

and `float(tensor_element)` is a device-to-host synchronisation. With a mean of 5.9 candidates
that is ~6 syncs per decision, plus a log_softmax over the full 251,048-row vocabulary. The
arithmetic accounts for the gap: a 368-token first forward is ~40 ms and a tie-break happens on
~15% of decisions, so the forwards explain ~46 ms of the 66.6 ms measured. Gathering every token
the decision needs in one indexing op and calling .tolist() once removes the rest without
touching a single number.
"""
import argparse
import gzip
import json
import os
import re
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

_OPT = re.compile(r"(?:^| )(\d+)=(\S+)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--maxlen", type=int, default=1024)
    a = ap.parse_args()

    import torch
    from tools.mirror_match import QwenScorer
    from lm.action_token import first_token, second_token, equivalent, to_scheme_b

    dec = []
    with gzip.open(a.data, "rt") as f:
        for line in f:
            d = json.loads(line)
            c = [t for _, t in _OPT.findall(d["prompt"].rsplit(":: ", 1)[-1])]
            if len(c) >= 2:
                dec.append((d["prompt"], c))
                if len(dec) >= a.n:
                    break

    sc = QwenScorer(a.ckpt, maxlen=a.maxlen)
    tk = sc.tk
    sc.model = sc.model.merge_and_unload()
    sc.model.eval()
    model, dev = sc.model, sc.model.device
    n_tie = [0]

    def prep(prompt):
        p = to_scheme_b(prompt) if sc.scheme_b else prompt
        return tk(p, add_special_tokens=False, truncation=True,
                  max_length=a.maxlen)["input_ids"]

    def tiebreaks(cands, heads):
        need = {}
        for h in set(heads):
            g = [i for i, x in enumerate(heads) if x == h]
            if len(g) > 1 and not all(equivalent(cands[i], cands[g[0]]) for i in g):
                need[h] = g
        return need

    # ---- A: round 2's best -----------------------------------------------------------------
    def cur(d):
        prompt, cands = d
        ids = prep(prompt)
        heads = [first_token(c) for c in cands]
        with torch.no_grad():
            out = model(input_ids=torch.tensor([ids], device=dev), use_cache=False,
                        logits_to_keep=1)
            lp1 = torch.log_softmax(out.logits[0, -1, :].float(), -1)
        hid = {h: tk.convert_tokens_to_ids(h) for h in set(heads)}
        res = [float(lp1[hid[h]]) if hid[h] is not None else -1e9 for h in heads]
        for h, grp in tiebreaks(cands, heads).items():
            with torch.no_grad():
                o2 = model(input_ids=torch.tensor([ids + [hid[h]]], device=dev),
                           use_cache=False, logits_to_keep=1)
                lp2 = torch.log_softmax(o2.logits[0, -1, :].float(), -1)
            for i in grp:
                t2 = second_token(cands[i]) if sc.scheme_b else None
                j = tk.convert_tokens_to_ids(t2) if t2 else None
                res[i] += float(lp2[j]) if j is not None else -1e9
        return res

    # ---- B: one gather, one sync -----------------------------------------------------------
    def gathered(d, use_kv=False):
        prompt, cands = d
        ids = prep(prompt)
        heads = [first_token(c) for c in cands]
        hid = {h: tk.convert_tokens_to_ids(h) for h in set(heads)}
        need = tiebreaks(cands, heads)
        cache = None
        with torch.no_grad():
            if use_kv and need:
                from transformers import DynamicCache
                out = model(input_ids=torch.tensor([ids], device=dev),
                            past_key_values=DynamicCache(), use_cache=True, logits_to_keep=1)
                cache = out.past_key_values
            else:
                out = model(input_ids=torch.tensor([ids], device=dev), use_cache=False,
                            logits_to_keep=1)
            row = out.logits[0, -1, :].float()
            lp1 = torch.log_softmax(row, -1)
            order = sorted(hid)
            want = torch.tensor([hid[h] if hid[h] is not None else 0 for h in order], device=dev)
            got = lp1[want].tolist()                      # ONE sync for the whole decision
        first = {h: (got[k] if hid[h] is not None else -1e9) for k, h in enumerate(order)}
        res = [first[h] for h in heads]
        if need:
            n_tie[0] += 1
        for h, grp in need.items():
            with torch.no_grad():
                if cache is not None:
                    o2 = model(input_ids=torch.tensor([[hid[h]]], device=dev),
                               past_key_values=cache, use_cache=True, logits_to_keep=1)
                    cache.crop(len(ids))
                else:
                    o2 = model(input_ids=torch.tensor([ids + [hid[h]]], device=dev),
                               use_cache=False, logits_to_keep=1)
                lp2 = torch.log_softmax(o2.logits[0, -1, :].float(), -1)
                toks = [second_token(cands[i]) if sc.scheme_b else None for i in grp]
                jj = [tk.convert_tokens_to_ids(t) if t else None for t in toks]
                vals = lp2[torch.tensor([j if j is not None else 0 for j in jj],
                                        device=dev)].tolist()
            for k, i in enumerate(grp):
                res[i] += vals[k] if jj[k] is not None else -1e9
        return res

    ref = [cur(d) for d in dec[:8]]

    def run(fn, label):
        try:
            got = [fn(d) for d in dec[:8]]
            worst = max(abs(x - y) for r, g in zip(ref, got) for x, y in zip(r, g))
        except Exception:
            print("  %-30s FAILED:" % label, flush=True)
            traceback.print_exc()
            return None
        for i in range(3):
            fn(dec[i])
        torch.cuda.synchronize(); t0 = time.time()
        for d in dec:
            fn(d)
        torch.cuda.synchronize()
        el = (time.time() - t0) / len(dec)
        print("  %-30s %6.1f ms/decision   max|diff| %.2e" % (label, 1000 * el, worst),
              flush=True)
        return el

    print("\n[merged + logits_to_keep, 120 real decisions]", flush=True)
    A = run(cur, "A round-2 best")
    n_tie[0] = 0
    B = run(gathered, "B + one gather/one sync")
    print("     tie-breaks fired on %d of %d decisions (%.1f%%)"
          % (n_tie[0], len(dec), 100.0 * n_tie[0] / len(dec)), flush=True)
    C = run(lambda d: gathered(d, use_kv=True), "C + KV cache for tie-break")
    print("\nbaseline today (unmerged, no logits_to_keep) = 134.1 ms/decision", flush=True)
    for lbl, v in (("A", A), ("B", B), ("C", C)):
        if v:
            print("  %s: %.1f ms -> %.2fx overall" % (lbl, 1000 * v, 0.1341 / v), flush=True)
    print("peak VRAM %.1f GiB" % (torch.cuda.max_memory_allocated() / 2**30), flush=True)


if __name__ == "__main__":
    main()
