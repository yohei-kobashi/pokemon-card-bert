#!/usr/bin/env python3
"""Convert DAgger records into the decoder's SFT format, and mix them with the base pool.

The two formats disagree on what the label indexes, which is a silent hazard:

    rerank   `chosen`      indexes the DEDUPED candidate list
    decoder  target        indexes the RENDERED MENU, which is NOT deduped

Using `chosen` as the decoder target would be off by however many duplicate options preceded it
-- a wrong label that trains cleanly and shows up only as a worse pilot. collect_dagger records
`menu_index` for exactly this, and records written before that field existed are refused rather
than silently mislabelled.

Mixing. The DAgger pool is small next to the base pool (tens of thousands against millions), so
a plain concatenation buries it. `--ratio` sets the DAgger SHARE of the output, by subsampling
the base pool to match; the DAgger records are never duplicated, because repeating the same few
thousand positions many times per epoch invites memorising them.
"""
import argparse
import gzip
import json
import random
import re


def n_options(prompt):
    return len(re.findall(r"(?:^| )(\d+)=", prompt.rsplit(":: ", 1)[-1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dagger", required=True)
    ap.add_argument("--base", default="", help="existing SFT jsonl.gz to mix in")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ratio", type=float, default=0.5,
                    help="DAgger share of the output (0.5 = half)")
    ap.add_argument("--errors-only", action="store_true",
                    help="keep only decisions the LM got wrong")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    recs, bad = [], 0
    with gzip.open(a.dagger, "rt") as f:
        for line in f:
            d = json.loads(line)
            mi = d.get("menu_index")
            if mi is None:
                bad += 1
                continue
            if a.errors_only and not d.get("lm_was_wrong"):
                continue
            prompt = "[ACT]\n" + d["state"]
            n = n_options(prompt)
            if n < 2 or mi >= n:
                bad += 1
                continue
            recs.append({"prompt": prompt, "target": str(mi), "src": "dagger"})
    if bad:
        print("skipped %d dagger records (no menu_index, or index outside the rendered menu)"
              % bad, flush=True)
    if not recs:
        raise SystemExit("no usable dagger records -- was the file written before menu_index "
                         "was added? Re-collect rather than guessing the label.")
    print("dagger usable: %d" % len(recs), flush=True)

    base = []
    if a.base:
        want = int(len(recs) * (1 - a.ratio) / max(1e-9, a.ratio))
        pool = 0
        with gzip.open(a.base, "rt") as f:
            for line in f:                        # reservoir: the file is far larger than `want`
                d = json.loads(line)
                if not d.get("target"):
                    continue
                pool += 1
                if len(base) < want:
                    base.append(d)
                else:
                    j = rng.randrange(pool)
                    if j < want:
                        base[j] = d
        print("base pool %d records, sampled %d for a %.0f%% dagger share"
              % (pool, len(base), 100 * a.ratio), flush=True)

    out = recs + base
    rng.shuffle(out)
    with gzip.open(a.out, "wt") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print("-> %s : %d records (%.1f%% dagger)"
          % (a.out, len(out), 100.0 * len(recs) / len(out)), flush=True)


if __name__ == "__main__":
    main()
