"""Actual (state, candidate) PAIR token lengths in the new data, and the batch cost.

max_len only matters when it truncates; the tokenizer pads to the LONGEST PAIR IN THE BATCH,
so the real cost driver is the p99/max of the pair length -- one long record drags the whole
batch's padding up. Print the distribution and what a pair-batch of N would actually cost, so
--max-len and --pair-batch are set from measurement rather than inherited from the 1024-token
full-glossary era.
"""
import gzip
import json
import random
import statistics
import sys

from transformers import AutoTokenizer

path, mdir = sys.argv[1], sys.argv[2]
n_rec = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
tok = AutoTokenizer.from_pretrained(mdir)

rows = []
with gzip.open(path, "rt") as f:
    for i, line in enumerate(f):
        if i % 37 == 0:
            rows.append(json.loads(line))
        if len(rows) >= n_rec:
            break

pair_lens, per_rec, cands = [], [], []
for r in rows:
    ls = [len(tok(r["state"], c)["input_ids"]) for c in r["candidates"]]
    pair_lens.extend(ls)
    per_rec.append(sum(ls))
    cands.append(len(r["candidates"]))

pair_lens.sort()


def q(xs, p):
    return xs[min(len(xs) - 1, int(p * len(xs)))]


print(f"records {len(rows)}  candidates/rec mean {statistics.mean(cands):.2f} "
      f"p90 {sorted(cands)[int(0.9 * len(cands))]} max {max(cands)}")
print(f"PAIR tokens  mean {statistics.mean(pair_lens):.0f}  p50 {q(pair_lens, .5)}  "
      f"p90 {q(pair_lens, .9)}  p99 {q(pair_lens, .99)}  max {max(pair_lens)}")
for ml in (256, 320, 384, 512, 640):
    trunc = sum(1 for x in pair_lens if x > ml)
    print(f"  max-len {ml:4d}: truncates {100.0 * trunc / len(pair_lens):5.2f}% of pairs")

# emulate the trainer's greedy packing: fill until pair_batch pairs, pad to that batch's max
for pb in (128, 192, 256, 384):
    random.Random(0).shuffle(rows)
    tot_pad = tot_real = nb = 0
    i = 0
    while i < len(rows):
        grp, np_ = [], 0
        while i < len(rows) and np_ < pb:
            grp.append(rows[i]); np_ += len(rows[i]["candidates"]); i += 1
        ls = [len(tok(r["state"], c)["input_ids"]) for r in grp for c in r["candidates"]]
        tot_real += sum(ls); tot_pad += len(ls) * max(ls); nb += 1
    print(f"  pair-batch {pb:3d}: {nb:4d} batches, padded tokens/real = "
          f"{tot_pad / tot_real:.2f}x, mean padded batch = {tot_pad / nb / 1000:.1f}k tokens")
