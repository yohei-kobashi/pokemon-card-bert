"""Score one or more reranker checkpoints on a FIXED eval split, with identity ablations.

Answers two questions the training log cannot:

  1. does the model actually USE each way we tell it which deck it is piloting?
     ``DECK[c1152x4,...]`` pins our deck exactly, so ``ID ME d_alakazam a_combo`` is
     REDUNDANT -- a segment the model ignores costs nothing to hide, and that is the
     measurement. ``swapME`` reproduces the corruption the v34 data actually shipped with
     (build_rerank._deck_names named the OPPONENT's deck in 50% of records), so its cost
     can be measured on a model instead of argued about.

  2. is a change to the embedding table (tools/init_domain_embeddings.py) survivable?
     Point it at several checkpoints and they are compared on the SAME rows.

The split is rebuilt with train_rerank's own seeds (reservoir 1234, shuffle 0, rows[:n]),
so the numbers line up with the run's ``[eval] top1``. ``--cache`` writes it out once.

    PYTHONPATH=cg-lib python tools/ablate_rerank.py --data <rerank.jsonl.gz> \
        --cache /root/out/eval2k.json --models /root/out/rerank_gte_v34,/root/out/x
"""
import argparse
import collections
import json
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

MASKS = (("full", {}),
         ("-DECK[]", {"drop_deck": True}),
         ("-ID ME", {"drop_identity": True}),
         ("-both", {"drop_deck": True, "drop_identity": True}),
         ("swapME", {"swap_identity": True}),
         ("swapDECK", {"swap_deck": True}))

_RE_TURN = re.compile(r"\bT(\d+)\.\d+")
_RE_DECKSEG = re.compile(r"^DECK\[[^\]]*\]")
# Aggregate top1 averages over every decision in a game, and the late game outnumbers the
# setup phase ~6:1. A segment that ONLY matters while the board is still empty can therefore
# look free in the mean while being decisive where it applies -- so bucket by turn.
BUCKETS = ((1, 2), (3, 5), (6, 10), (11, 999))


def turn_of(state):
    m = _RE_TURN.search(state)
    return int(m.group(1)) if m else 0


def bucket_of(t):
    for lo, hi in BUCKETS:
        if lo <= t <= hi:
            return f"T{lo}-{hi if hi < 999 else '+'}"
    return "T?"


def swap_deck_pool(rows):
    """For each row, a DECK[...] literal belonging to a DIFFERENT deck.

    ``-DECK[]`` deletes the segment, which is also a shift OFF the training distribution, so
    its cost conflates 'the model reads this' with 'the model has never seen it missing'.
    Substituting another real deck's list keeps the shape and the token statistics identical
    and changes only the CONTENT -- if the answer does not move, the content is unused."""
    by_deck = {}
    for r in rows:
        m = _RE_DECKSEG.match(r["state"])
        if m:
            by_deck.setdefault(r.get("deck") or m.group(0), m.group(0))
    keys = sorted(by_deck)
    if len(keys) < 2:
        return {}
    return {k: by_deck[keys[(i + 1) % len(keys)]] for i, k in enumerate(keys)}


def eval_rows(args):
    if args.cache and os.path.exists(args.cache):
        rows = json.load(open(args.cache))
        print(f"eval split from cache: {len(rows)} rows", flush=True)
        return rows
    from train_rerank import read_rows
    # must mirror the RUN's sampling exactly, cap_matchup included -- a balanced run holds
    # out rows[:eval_n] of the BALANCED pool, which is a different set from the uniform one
    rows = read_rows(args.data, args.max_samples, cap_matchup=args.cap_matchup)
    random.Random(0).shuffle(rows)                    # train_rerank does exactly this
    rows = rows[:args.eval_n]
    if args.cache:
        json.dump(rows, open(args.cache, "w"))
    print(f"eval split rebuilt: {len(rows)} rows -> {args.cache}", flush=True)
    return rows


def main():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from lm.serialize import mask_segments

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="", help="comma-separated checkpoint dirs; omit to "
                    "only BUILD --cache (CPU-only, safe to run beside a training job)")
    ap.add_argument("--data", default="")
    ap.add_argument("--cache", default="")
    ap.add_argument("--max-samples", type=int, default=1200000)
    ap.add_argument("--cap-matchup", type=int, default=0,
                    help="mirror the run's --cap-matchup so the held-out split matches")
    ap.add_argument("--eval-n", type=int, default=2000)
    ap.add_argument("--pair-batch", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=640)
    ap.add_argument("--by-turn", action="store_true",
                    help="also break every mask down by turn bucket")
    args = ap.parse_args()

    rows = eval_rows(args)
    if not args.models:
        return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pool = swap_deck_pool(rows)
    bkt = [bucket_of(turn_of(r["state"])) for r in rows]
    order = [f"T{lo}-{hi if hi < 999 else '+'}" for lo, hi in BUCKETS]
    if args.by_turn:
        c = collections.Counter(bkt)
        print("rows per bucket: " + "  ".join(f"{b} {c.get(b, 0)}" for b in order), flush=True)

    def masked(r, mk):
        if mk.get("swap_deck"):
            m = _RE_DECKSEG.match(r["state"])
            rep = pool.get(r.get("deck") or m.group(0)) if m else None
            return (rep + r["state"][m.end():]) if rep else r["state"]
        return mask_segments(r["state"], **mk) if mk else r["state"]

    print(f"\n{'checkpoint':34s} " + " ".join(f"{n:>9s}" for n, _ in MASKS), flush=True)
    for mdir in args.models.split(","):
        tok = AutoTokenizer.from_pretrained(mdir)
        model = AutoModelForSequenceClassification.from_pretrained(
            mdir, trust_remote_code=True, dtype=torch.bfloat16).to(dev).eval()
        cells, per_bucket = [], {}
        for _name, mk in MASKS:
            hit = collections.Counter()
            tot = collections.Counter()
            i = 0
            with torch.no_grad():
                while i < len(rows):
                    grp, idx, npairs = [], [], 0
                    while i < len(rows) and npairs < args.pair_batch:
                        grp.append(rows[i]); idx.append(i)
                        npairs += len(rows[i]["candidates"]); i += 1
                    pairs, owner = [], []
                    for ri, r in enumerate(grp):
                        st = masked(r, mk)
                        for c in r["candidates"]:
                            pairs.append([st, c]); owner.append(ri)
                    enc = tok(pairs, padding=True, truncation="only_first",
                              max_length=args.max_len, return_tensors="pt").to(dev)
                    lg = model(**enc).logits.squeeze(-1).float()
                    per = [[] for _ in grp]
                    for k, ri in enumerate(owner):
                        per[ri].append(lg[k])
                    for j, (r, s) in zip(idx, zip(grp, per)):
                        ok = int(int(torch.stack(s).argmax()) == r["chosen"])
                        hit["all"] += ok; tot["all"] += 1
                        hit[bkt[j]] += ok; tot[bkt[j]] += 1
            cells.append(100.0 * hit["all"] / max(1, tot["all"]))
            per_bucket[_name] = {b: (100.0 * hit[b] / tot[b]) if tot[b] else float("nan")
                                 for b in order}
        print(f"{os.path.basename(mdir):34s} " + " ".join(f"{c:8.1f}%" for c in cells),
              flush=True)
        if args.by_turn:
            for b in order:
                print(f"  {b:32s} " + " ".join(f"{per_bucket[n][b]:8.1f}%"
                                               for n, _ in MASKS), flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
