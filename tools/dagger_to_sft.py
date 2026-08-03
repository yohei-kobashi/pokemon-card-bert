#!/usr/bin/env python3
"""Convert DAgger records into the decoder's SFT format, and mix them with the base pool.

The two formats disagree on what the label indexes, which is a silent hazard:

    rerank   `chosen`      indexes the DEDUPED candidate list
    decoder  target        indexes the RENDERED MENU, which is NOT deduped

Using `chosen` as the decoder target would be off by however many duplicate options preceded it
-- a wrong label that trains cleanly and shows up only as a worse pilot. Measured on the 126,090
records from instance1's round-1 collection: 26.5% of them HAD a duplicate removed, so `chosen`
would have mislabelled a quarter of the data.

collect_dagger records `menu_index` for exactly this. Files written before that field existed are
not refused, because the label is RECOVERABLE without guessing: the rendered menu is part of the
prompt, so `candidates[chosen]` can be matched against it by exact string equality. On those same
126,090 records that resolves 92.3% to a unique index and 7.7% to several identical option texts
-- and identical text means the same action from the model's point of view, so any of them is a
correct label. Nothing matched zero options. A record that fails to match is dropped and counted,
never guessed.

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


_RE_OPT = re.compile(r"(?:^| )(\d+)=(\S+)")


def menu_options(prompt):
    """-> the rendered menu as a list of option texts, or None if it does not parse.

    A parse is only accepted when the printed indices are exactly 0..k-1 in order; anything else
    means the tail being read is not the menu, and a label derived from it would be arbitrary.
    """
    opts = _RE_OPT.findall(prompt.rsplit(":: ", 1)[-1])
    if [int(i) for i, _ in opts] != list(range(len(opts))):
        return None
    return [t for _, t in opts]


def n_options(prompt):
    return len(re.findall(r"(?:^| )(\d+)=", prompt.rsplit(":: ", 1)[-1]))


def recover_menu_index(prompt, candidates, chosen):
    """Locate the chosen candidate in the rendered menu by exact text match. -> index or None."""
    texts = menu_options(prompt)
    if texts is None or not (0 <= chosen < len(candidates)):
        return None
    want = candidates[chosen]
    for i, t in enumerate(texts):
        if t == want:
            return i          # duplicates render identically, so the first is as right as any
    return None


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
    recs, bad, recovered = [], 0, 0
    with gzip.open(a.dagger, "rt") as f:
        for line in f:
            d = json.loads(line)
            if a.errors_only and not d.get("lm_was_wrong"):
                continue
            prompt = "[ACT]\n" + d["state"]
            mi = d.get("menu_index")
            if mi is None:
                mi = recover_menu_index(prompt, d.get("candidates") or [], d.get("chosen", -1))
                if mi is None:
                    bad += 1
                    continue
                recovered += 1
            n = n_options(prompt)
            if n < 2 or mi >= n:
                bad += 1
                continue
            recs.append({"prompt": prompt, "target": str(mi), "src": "dagger"})
    if recovered:
        print("recovered menu_index for %d records by matching the candidate text against the "
              "rendered menu" % recovered, flush=True)
    if bad:
        print("skipped %d dagger records (label not recoverable, or index outside the menu)"
              % bad, flush=True)
    if not recs:
        raise SystemExit("no usable dagger records -- neither menu_index nor a recoverable "
                         "candidate/menu match. Re-collect rather than guessing the label.")
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
