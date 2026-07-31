#!/usr/bin/env python3
"""Full-coverage candidate scoring for the Qwen3.5 teacher.

`sft_teacher.eval_top1` scores only decisions with <= 10 options, because there the answer is a
single token and the whole decision is one argmax.  That silently drops 20.5% of the held-out set
(819/4000 measured), and the dropped rows are exactly the ones with MANY options -- the hard
ones -- so its number is biased high.

This scores every decision by giving each candidate index its full token-sequence log-probability
under teacher forcing, which also produces the object the distillation actually needs: a
probability distribution over the LEGAL candidate set.

Scoring, per decision:

    seqs[k] = tokenize(str(k))                       for k in 0 .. n_options-1
    logP(k) = sum_d  log softmax(model(prompt + seqs[k][:d]))[seqs[k][d]]
    dist    = softmax over k of logP(k)              <- renormalised onto legal moves only

Cost is NOT one forward per candidate: candidates share prefixes, so the number of forwards is
the number of distinct prefixes.  n <= 10 needs 1 forward (identical to today).  n = 51 needs 6.
Measured option counts are p50 5 / p90 13 / p99 26 / max 51, so the average is ~1.3 forwards.

`legal_mass` is reported because restricting to legal indices HIDES a broken teacher: if the model
wanted to emit prose, the restricted softmax still returns a clean-looking distribution.  Low
legal mass means that row's target is not trustworthy and should not be distilled.
"""
import argparse
import gzip
import json
import math
import re
import time

OPT_RE = re.compile(r"(?:^| )(\d+)=")


def n_options(prompt):
    """How many numbered options the rendered menu offers (same rule as sft_teacher)."""
    return len(OPT_RE.findall(prompt.rsplit(":: ", 1)[-1]))


def load_pairs(path, limit, skip=0):
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
    return P, C


def index_seqs(tk, n, cache={}):
    """Token-id sequence for each index string "0".."n-1".

    Tokenized STANDALONE with no leading space, because that is how TRL tokenizes the completion
    of a prompt-completion pair -- prompt and completion are encoded separately and concatenated,
    so the answer the model was trained to emit is exactly these ids.
    """
    out = []
    for k in range(n):
        if k not in cache:
            cache[k] = tuple(tk(str(k), add_special_tokens=False)["input_ids"])
        out.append(cache[k])
    return out


def score_decision(model, tk, torch, prompt, n, maxlen, prefix_cache_stats=None):
    """-> (logP per candidate, legal_mass, n_forwards).

    Batch size 1 throughout: no padding, so nothing depends on whether the DeltaNet layers honour
    an attention mask over pad positions.  `--pad-check` measures that separately.
    """
    seqs = index_seqs(tk, n)
    base = tk(prompt, add_special_tokens=False, truncation=True, max_length=maxlen)["input_ids"]
    dev = model.device
    lsm = torch.nn.functional.log_softmax

    cont = {}          # prefix (tuple of ids) -> log-softmax vector at the next position
    fwd = 0
    depth = max(len(s) for s in seqs)
    prefix_lp = {(): 0.0}
    legal_mass = None
    for d in range(depth):
        for p in sorted({s[:d] for s in seqs if len(s) > d}):
            if p in cont:
                continue
            ids = torch.tensor([base + list(p)], device=dev)
            with torch.no_grad():
                lg = model(input_ids=ids).logits[0, -1, :].float()
            cont[p] = lsm(lg, dim=-1)
            fwd += 1
            if d == 0:
                # probability the model puts on ANY legal first token, before renormalisation
                first = sorted({s[0] for s in seqs})
                legal_mass = float(cont[p][first].exp().sum())
        for s in seqs:
            if len(s) > d:
                prefix_lp[s[:d + 1]] = prefix_lp[s[:d]] + float(cont[s[:d]][s[d]])
    if prefix_cache_stats is not None:
        prefix_cache_stats.append(fwd)
    return [prefix_lp[s] for s in seqs], legal_mass, fwd


def softmax(xs, temp=1.0):
    m = max(xs)
    e = [math.exp((x - m) / temp) for x in xs]
    z = sum(e)
    return [v / z for v in e]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen3.5-9B-Base",
                    help="base model, or an adapter directory saved by sft_teacher")
    ap.add_argument("--adapter", default=None, help="LoRA adapter directory to load on top")
    ap.add_argument("--data", default="/root/ptcg/repo/data/sft/teacher_0730_index.jsonl.gz")
    ap.add_argument("--eval-n", type=int, default=4000,
                    help="held-out slice: the FRONT of the file, which training skipped")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N held-out rows")
    ap.add_argument("--maxlen", type=int, default=1024)
    ap.add_argument("--dump", default=None, help="write per-decision candidate distributions here")
    ap.add_argument("--temp", type=float, default=1.0, help="temperature for the dumped dist")
    ap.add_argument("--pad-check", type=int, default=0,
                    help="also score N rows with LEFT-PADDED batching and report disagreement")
    a = ap.parse_args()

    t0 = time.time()
    from unsloth import FastLanguageModel          # noqa: E402  (must precede transformers)
    import torch                                   # noqa: E402

    model, tok = FastLanguageModel.from_pretrained(
        model_name=a.model, max_seq_length=a.maxlen,
        load_in_4bit=False, load_in_16bit=True, full_finetuning=False)
    if a.adapter:
        from peft import PeftModel                 # noqa: E402
        model = PeftModel.from_pretrained(model, a.adapter)
        print("[adapter] %s" % a.adapter, flush=True)
    model.eval()
    # Qwen3.5 is a VLM: `tok` is a PROCESSOR, and a positional call binds the first argument to
    # `images`.  Use the underlying text tokenizer.
    tk = getattr(tok, "tokenizer", tok)
    if tk.pad_token is None:
        tk.pad_token = tk.eos_token
    print("[load] %.1fs" % (time.time() - t0), flush=True)

    P, C = load_pairs(a.data, a.eval_n)
    if a.limit:
        P, C = P[:a.limit], C[:a.limit]

    # report the tokenization of the index strings once -- everything downstream depends on it
    ml = {}
    for k in range(52):
        ml.setdefault(len(index_seqs(tk, k + 1)[k]), []).append(k)
    print("[index tokens] " + " | ".join("%d tok: %d values (%s..)" % (L, len(v), v[0])
                                         for L, v in sorted(ml.items())), flush=True)

    dump = gzip.open(a.dump, "wt") if a.dump else None
    ok = tot = skipped = 0
    ok_small = tot_small = ok_big = tot_big = 0
    masses, fwds = [], []
    for i, (p, c) in enumerate(zip(P, C)):
        n = n_options(p)
        if n < 2 or not c.isdigit() or int(c) >= n:
            skipped += 1
            continue
        lp, mass, _ = score_decision(model, tk, torch, p, n, a.maxlen, fwds)
        best = max(range(n), key=lambda k: lp[k])
        hit = int(best == int(c))
        ok += hit
        tot += 1
        masses.append(mass)
        if n <= 10:
            ok_small += hit
            tot_small += 1
        else:
            ok_big += hit
            tot_big += 1
        if dump:
            dump.write(json.dumps({"i": i, "n": n, "target": int(c),
                                   "logp": [round(x, 5) for x in lp],
                                   "dist": [round(x, 6) for x in softmax(lp, a.temp)],
                                   "legal_mass": round(mass, 6)}) + "\n")
        if tot % 250 == 0:
            print("  %d/%d  top1 %.2f%%  legal_mass %.3f  %.1f fwd/dec  %.0fs"
                  % (tot, len(P), 100.0 * ok / tot, sum(masses) / len(masses),
                     sum(fwds) / len(fwds), time.time() - t0), flush=True)
    if dump:
        dump.close()

    print("\n[FULL COVERAGE] top1 %d/%d = %.2f%%   (unscorable %d)"
          % (ok, tot, 100.0 * ok / max(1, tot), skipped), flush=True)
    print("  <=10 options  %d/%d = %.2f%%      <- what eval_top1 reported"
          % (ok_small, tot_small, 100.0 * ok_small / max(1, tot_small)), flush=True)
    print("  >10 options   %d/%d = %.2f%%      <- what it dropped"
          % (ok_big, tot_big, 100.0 * ok_big / max(1, tot_big)), flush=True)
    print("  legal_mass    mean %.4f  min %.4f  (share of vocab probability on a legal index)"
          % (sum(masses) / max(1, len(masses)), min(masses) if masses else 0), flush=True)
    print("  forwards/dec  mean %.2f" % (sum(fwds) / max(1, len(fwds))), flush=True)

    if a.pad_check:
        # eval_top1 batches with LEFT padding.  3 of every 4 Qwen3.5 layers are linear attention
        # carrying a recurrent state; if those kernels ignore the attention mask, pad tokens enter
        # the state and every batched number is wrong.  Compare against the batch-1 scores above.
        tk.padding_side = "left"
        # Only n <= 10 rows: there every candidate is ONE token, so the batched pass's
        # first-token log-prob and the batch-1 sequence log-prob are the same quantity and any
        # difference is padding.  On multi-token rows they differ legitimately, which would swamp
        # the measurement.
        rows = [(p, c) for p, c in zip(P, C)
                if 2 <= n_options(p) <= 10 and c.isdigit() and int(c) < n_options(p)][:a.pad_check]
        agree = 0
        dmax = 0.0
        for j in range(0, len(rows), 8):
            b = rows[j:j + 8]
            enc = tk([p for p, _ in b], return_tensors="pt", padding=True,
                     truncation=True, max_length=a.maxlen).to(model.device)
            with torch.no_grad():
                out = model(**enc).logits[:, -1, :].float()
            lsm = torch.nn.functional.log_softmax
            for r, (p, _) in enumerate(b):
                n = n_options(p)
                seqs = index_seqs(tk, n)
                pad = lsm(out[r], dim=-1)
                one, _, _ = score_decision(model, tk, torch, p, n, a.maxlen)
                f_pad = [float(pad[s[0]]) for s in seqs]
                f_one = [float(x) for x in one]
                agree += int(max(range(n), key=lambda k: f_pad[k])
                             == max(range(n), key=lambda k: f_one[k]))
                dmax = max(dmax, max(abs(x - y) for x, y in zip(f_pad, f_one)))
        print("\n[pad-check] left-padded batch vs batch-1 on %d single-token rows: "
              "argmax agree %d, max |dlogp| %.4f" % (len(rows), agree, dmax), flush=True)
        print("  These are the SAME quantity computed two ways, so agree must be %d and |dlogp| "
              "~0.  Anything else means left padding enters the DeltaNet recurrent state and "
              "every batched number this session -- including [eval BEFORE] 43.29%% -- is "
              "measured wrong." % len(rows), flush=True)

    print("[total] %.1f min" % ((time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
