#!/usr/bin/env python3
"""Score a reranker checkpoint on ATTACH decisions whose value was MEASURED, not assumed.

WHY NOT THE MIRROR SCREEN. The screen is the only instrument that answers "is it stronger", but
its resolution is poor: the same loop produced medians 35.5 / 41.0 / 32.0 on three consecutive
rounds, a ~9pt swing, so it cannot see a change that touches 8% of decisions. It stays the
confirmation; it cannot be the discriminator.

WHY NOT TOP1 EITHER. Top1 against engine_v2 measures agreement with a teacher that is itself
only 48% of the way to the best attach (`attach-value-measured`), and it scores a near-miss and
a blunder the same. The metric here is the one the +0.0586 / +0.1221 numbers are on:

    E[Q(model's pick)] - E[Q(the alternatives)]

on the same +/-1 scale, against three reference points measured on the SAME decisions:

    engine   what engine_v2's own pick is worth   (published: +0.0586 +- 0.0058)
    best     what the measured-best pick is worth (published: +0.1221 +- 0.0074)
    chance   what picking uniformly is worth      (0 by construction)

A model that has learned nothing about attach lands at chance; one that has learned to imitate
engine_v2 lands near `engine`; the value-margin term is trying to move it toward `best`.

The held-out file must be built with `--uniform` (no headroom targeting), or the numbers
describe the cells that were sampled rather than the decisions the pilot actually faces.
"""
import argparse
import collections
import gzip
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)


def edge(q, pick, valued):
    """Q(pick) minus the mean Q of the other VALUED candidates, or None if undefined."""
    others = [q[i] for i in valued if i != pick]
    if not others or q[pick] is None:
        return None
    return q[pick] - sum(others) / len(others)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="checkpoint dir")
    ap.add_argument("--data", required=True, help="attach_label.py output (--uniform)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pair-batch", type=int, default=192)
    ap.add_argument("--max-len", type=int, default=768)
    a = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    rows = []
    with gzip.open(a.data, "rt") as f:
        for line in f:
            d = json.loads(line)
            v = [i for i, x in enumerate(d.get("qvals") or []) if x is not None]
            if len(v) >= 2:
                d["_valued"] = v
                rows.append(d)
            if a.limit and len(rows) >= a.limit:
                break
    if not rows:
        raise SystemExit("no valued rows in %s" % a.data)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model, truncation_side="left")
    model = AutoModelForSequenceClassification.from_pretrained(
        a.model, num_labels=1, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32)
    model.to(dev).eval()

    picks = []
    pairs, owner = [], []

    @torch.no_grad()
    def flush():
        if not pairs:
            return
        enc = tok(pairs, padding=True, truncation="only_first", max_length=a.max_len,
                  return_tensors="pt").to(dev)
        lg = model(**enc).logits.squeeze(-1).float().tolist()
        per = collections.defaultdict(list)
        for k, ri in enumerate(owner):
            per[ri].append(lg[k])
        for ri, s in per.items():
            r = rows[ri]
            # the model ranks EVERY candidate; the pick only counts if it is a valued one,
            # because a decision where it plays something other than an attach is not a
            # judgement about attach targets at all
            best = max(range(len(s)), key=lambda i: s[i])
            picks.append((ri, best))
        pairs.clear()
        owner.clear()

    for ri, r in enumerate(rows):
        for c in r["candidates"]:
            pairs.append([r["state"], c])
            owner.append(ri)
        if len(pairs) >= a.pair_batch:
            flush()
    flush()

    model_e, eng_e, best_e, paired = [], [], [], []
    top1 = off_attach = 0
    by_deck = collections.defaultdict(list)
    for ri, best in picks:
        r = rows[ri]
        q, v = r["qvals"], r["_valued"]
        if q[best] is None:
            off_attach += 1
            continue
        e = edge(q, best, v)
        if e is None:
            continue
        model_e.append(e)
        by_deck[r["deck"]].append(e)
        eg = r.get("engine_chosen")
        if eg is not None and q[eg] is not None:
            ee = edge(q, eg, v)
            if ee is not None:
                eng_e.append(ee)
                # model and engine judged the SAME decision, so the difference is paired and
                # its error is far below the marginal errors -- the between-decision variance
                # (some decisions simply have more at stake) cancels
                paired.append(e - ee)
        be = edge(q, r["chosen"], v)
        if be is not None:
            best_e.append(be)
        top1 += (best == r["chosen"])

    def line(name, xs):
        if not xs:
            print("  %-22s (none)" % name)
            return
        sd = statistics.pstdev(xs)
        print("  %-22s %+.4f +- %.4f   n %d" % (name, sum(xs) / len(xs), sd / len(xs) ** 0.5,
                                                len(xs)))

    print("model %s\ndata  %s  (%d decisions)" % (a.model, a.data, len(rows)))
    print("\n=== value of the chosen attach target, +/-1 scale ===")
    line("MODEL", model_e)
    line("engine_v2", eng_e)
    line("measured best", best_e)
    line("MODEL - engine (paired)", paired)
    print("  %-22s %.1f%%" % ("top1 vs measured best", 100.0 * top1 / max(1, len(model_e))))
    print("  %-22s %d (%.1f%%) -- scored a non-attach above every attach"
          % ("off-attach picks", off_attach, 100.0 * off_attach / max(1, len(picks))))
    if eng_e and model_e and best_e:
        m, e, b = (sum(x) / len(x) for x in (model_e, eng_e, best_e))
        print("\n  of the available edge (chance 0 -> best): model %.0f%%, engine %.0f%%"
              % (100.0 * m / max(1e-9, b), 100.0 * e / max(1e-9, b)))
        print("  of the HEADROOM above engine (engine -> best): %+.0f%%"
              % (100.0 * (m - e) / max(1e-9, b - e)))
    print("\nworst 8 decks for the model:")
    for d, xs in sorted(by_deck.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))[:8]:
        print("  %-24s %+.4f  n%d" % (d, sum(xs) / len(xs), len(xs)))


if __name__ == "__main__":
    main()
