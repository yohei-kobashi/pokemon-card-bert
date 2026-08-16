"""Undo --deck-shuffle in an already-built rerank file: sort DECK[...] by card id.

The ablation arm for "did per-decision DECK[] permutation cost us anything" only needs a FIXED
order, and sorting gives one without re-running build_rerank (which would re-render 12.5M
states and re-generate nothing new). Everything else in the record -- states, candidates,
labels, which side, remaining-mode contents -- is byte-identical, so the arm differs from the
control in exactly one property.

Note this leaves deck-mode=remaining intact: the multiset still changes every decision, it
just no longer changes ORDER for a fixed multiset.
"""
import gzip
import json
import multiprocessing as mp
import os
import re
import sys

RE_DECK = re.compile(r"^DECK\[([^\]]*)\]")


def sort_deck(state):
    m = RE_DECK.match(state)
    if not m:
        return state
    body = m.group(1)
    if not body:
        return state
    entries = body.split(",")
    def key(e):
        return int(e[1:].split("x")[0])
    return "DECK[" + ",".join(sorted(entries, key=key)) + "]" + state[m.end():]


def work(job):
    src, dst, lo, hi = job
    n = 0
    with gzip.open(src, "rt") as fi, gzip.open(dst, "wt") as fo:
        for i, line in enumerate(fi):
            if i < lo or i >= hi:
                continue
            r = json.loads(line)
            r["state"] = sort_deck(r["state"])
            fo.write(json.dumps(r, separators=(",", ":")) + "\n")
            n += 1
    return dst, n


def main():
    src, dst = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    with gzip.open(src, "rt") as f:
        total = sum(1 for _ in f)
    step = (total + workers - 1) // workers
    jobs = [(src, "%s.part%02d" % (dst, k), k * step, (k + 1) * step)
            for k in range(workers)]
    print("%s: %d rows -> %d shards" % (os.path.basename(src), total, len(jobs)), flush=True)
    with mp.Pool(workers) as pool:
        done = pool.map(work, jobs)
    got = 0
    with gzip.open(dst, "wt") as fo:
        for path, n in done:
            with gzip.open(path, "rt") as fi:
                for line in fi:
                    fo.write(line)
            got += n
            os.remove(path)
    print("wrote %s  %d rows (in %d)" % (dst, got, total), flush=True)
    assert got == total, "row count changed"


if __name__ == "__main__":
    main()
