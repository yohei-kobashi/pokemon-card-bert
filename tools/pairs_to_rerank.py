"""DPO pairs -> the listwise {state, candidates, chosen} shape the vocab tools read.

The deploy pipeline (sweep_vocab_rerank -> prune_vocab_rerank -> quant_weightonly_rerank) takes
a rerank jsonl.gz, and every one we have is a JULY file in the v2/v36 prompt formats. The
mirror-RL champion is trained on DUSK_FMT, whose prompt is a different length and a different
token set, so pruning it against v36 records would keep the wrong rows -- and the failure is
silent: an id that was pruned away comes back as [UNK], which is a legal token.

The pairs are the right source because they hold the EXACT strings the model was trained on:
`prompt` is serialize_stateless under DUSK_FMT and `cw`/`cl` are encode_option outputs. What
they do not hold is the full menu -- a pair is the top-2 by margin -- so a record here has two
candidates, not n. That is fine for the two consumers: the vocab sweep unions token ids (and
takes the whole fleet's card/attack/deck tokens from decks/ separately, so unseen cards are
covered regardless), and the prune/quant verifiers compare pruned-vs-original logits on real
(state, candidate) pairs, where two candidates per state is a comparison like any other.

`chosen` is 0 by construction: cw is the preferred side. Nothing downstream of here trains on
it, but emitting a wrong index would be a trap for whatever reads this next.

    python tools/pairs_to_rerank.py --pairs '/root/mrl*_pairs*.jsonl.gz' --out /root/dusk.rerank.jsonl.gz
"""
import argparse
import glob
import gzip
import json


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True, help="glob or comma-separated jsonl.gz pair files")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    paths = []
    for part in a.pairs.split(","):
        paths.extend(sorted(glob.glob(part.strip())))
    if not paths:
        raise SystemExit("no pair files matched %r" % a.pairs)

    n_in = n_out = 0
    seen = set()
    with gzip.open(a.out, "wt", encoding="utf-8") as w:
        for p in paths:
            k = 0
            with gzip.open(p, "rt", encoding="utf-8") as fh:
                for line in fh:
                    r = json.loads(line)
                    n_in += 1
                    st, cw, cl = r.get("prompt"), r.get("cw"), r.get("cl")
                    if not st or not cw or not cl:
                        continue
                    # Rounds re-collect from the same seeds, so the same decision recurs across
                    # files. Deduping here is not cosmetic: the sweep tokenizes every unique
                    # state anyway, and the verifiers sample the first --n records, which would
                    # otherwise all come from round 1.
                    key = (hash(st), cw, cl)
                    if key in seen:
                        continue
                    seen.add(key)
                    w.write(json.dumps({"state": st, "candidates": [cw, cl], "chosen": 0}) + "\n")
                    n_out += 1
                    k += 1
            print("  %-40s %d" % (p.rsplit("/", 1)[-1], k), flush=True)
    print("%d pairs -> %d records (%d duplicates dropped) -> %s"
          % (n_in, n_out, n_in - n_out, a.out))


if __name__ == "__main__":
    main()
