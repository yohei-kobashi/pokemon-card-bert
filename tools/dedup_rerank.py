#!/usr/bin/env python3
"""Collapse equivalent candidates in an ALREADY BUILT reranker/DAgger pool.

`tools/build_rerank.py` now dedups by the act rather than by the rendered text, but the existing
pools were written before that and regenerating them means replaying the games. Nothing in a
record depends on the games, though: the candidate list and the chosen index are the whole
sample, so the same result is reachable by rewriting the file.

What it removes, measured on data/rerank/v39_0731.rerank.jsonl.gz:

    records with a negative that is the SAME ACT as the positive   17.17%
      card 48,811 | facedown 18,657 | energy 1,220
    candidates per decision                                        5.84 -> 5.20
    records that collapse to a single candidate (no choice left)   5.65%

The last line is why records are DROPPED rather than kept with one candidate: a listwise record
with a single option teaches nothing, and at inference the reranker would spend a forward pass
discovering it had no decision to make.

SECOND PASS (2026-08-02): board slots the PROMPT renders identically. Three copies of one Basic
on the bench at full HP with nothing attached are one attach target written three ways, and
engine_v2's pick between them is arbitrary -- measured over 80,000 base-pool records, `attach`
carries a top1 ceiling of 90.2% and `evolve` 85.3% for a PERFECT model, which is label noise no
amount of training removes. Per `attach-decisions-at-chance` attach was already the worst
decision kind by a wide margin, so this is the one place where cleaning the label is worth more
than any extra data.

The descriptors are recovered from each record's own `state` -- the rendered prompt is what the
model reads, so "the same to the model" is decided by exactly the text it was shown. The live
path in lm/agent reads them from the observation instead; the two were checked to agree on
2,190 decisions across four decks, 0 disagreements.
"""
import argparse
import gzip
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib")):
    if p not in sys.path:
        sys.path.insert(0, p)

from lm.action_token import dedup_options       # noqa: E402


def dedup(cands, chosen, state=None):
    """-> (new candidates, remap) or (None, None) when no choice survives.

    `remap` turns any old candidate index into its new one, so every index a record carries --
    `chosen`, and `lm_chosen` on DAgger rows -- goes through the SAME mapping. Remapping them
    with two separate lookups is how one of them ends up naming a different move.
    """
    if not 0 <= chosen < len(cands):
        return None, None
    keep, pos, keys = dedup_options(cands, state=state)
    if len(keep) < 2:
        return None, None
    at = {keys[p]: i for i, p in enumerate(pos)}
    return keep, [at.get(k) for k in keys]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-board", action="store_true",
                    help="text-only collapse, i.e. reproduce the first pass exactly")
    a = ap.parse_args()

    n = kept = 0
    before = after = 0
    with gzip.open(a.inp, "rt") as f, gzip.open(a.out, "wt") as g:
        for line in f:
            d = json.loads(line)
            c, k = d.get("candidates"), d.get("chosen")
            if not c or k is None:
                continue
            n += 1
            before += len(c)
            nc, remap = dedup(c, k, None if a.no_board else d.get("state"))
            if nc is None:
                continue
            # `lm_chosen` is an index into the OLD candidate list, so it goes through the same
            # remap or it silently starts naming a different move. A collapsed pair means the
            # LM and the engine picked the SAME act, so `lm_was_wrong` has to be recomputed --
            # otherwise records land in the pool flagged as errors that no longer are.
            if d.get("lm_chosen") is not None:
                d["lm_chosen"] = remap[d["lm_chosen"]] \
                    if 0 <= d["lm_chosen"] < len(remap) else None
                d["lm_was_wrong"] = (d["lm_chosen"] != remap[k])
            # `menu_index` points into the UN-deduped rendered menu and is what the decoder's
            # target indexes, so it must survive this untouched -- it is not an index into
            # `candidates` and remapping it here would corrupt the decoder pool.
            d["candidates"], d["chosen"] = nc, remap[k]
            after += len(nc)
            kept += 1
            g.write(json.dumps(d, ensure_ascii=False) + "\n")
    print("%s -> %s" % (a.inp, a.out), flush=True)
    print("  records %d -> %d (%.2f%% dropped: a single act was left)"
          % (n, kept, 100.0 * (n - kept) / max(1, n)), flush=True)
    print("  candidates per record %.2f -> %.2f (%.1f%% fewer forwards at inference)"
          % (before / max(1, n), after / max(1, kept),
             100.0 * (1 - (after / max(1, kept)) / (before / max(1, n)))), flush=True)


if __name__ == "__main__":
    main()
