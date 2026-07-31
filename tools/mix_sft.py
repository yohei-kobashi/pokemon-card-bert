"""Mix build_sft imitation data with search-distilled contrastive data into one
SFT set with THREE consistent multi-task targets (no format collision):
  [ACT]     -> move                       (imitation act + contrastive act)
  [REASON]  -> realized future + outcome  (build_sft reason)
  [COMPARE] -> A/B rollouts + verdict      (contrastive)
Contrastive samples are identified structurally (target 'A: ... => CHOOSE ...'),
so data produced by the currently-running kernels (still tagged [REASON]/reason)
is RE-TAGGED to [COMPARE]/compare here. Idempotent.
"""
import argparse, glob, gzip, json, os, random


def is_compare(d):
    t = d.get("target", "")
    return t.startswith("A: ") and "=> CHOOSE " in t


def retag(d):
    if is_compare(d) and d.get("mode") != "compare":
        d["mode"] = "compare"
        if d["prompt"].startswith("[REASON]\n"):
            d["prompt"] = "[COMPARE]\n" + d["prompt"][len("[REASON]\n"):]
        elif not d["prompt"].startswith("[COMPARE]"):
            d["prompt"] = "[COMPARE]\n" + d["prompt"].split("\n", 1)[-1]
    return d


def read(paths):
    for p in paths:
        op = gzip.open if p.endswith(".gz") else open
        with op(p, "rt") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imitation", nargs="*", default=[])
    ap.add_argument("--contrastive", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from collections import Counter
    modes = Counter()
    rows = []
    for d in read(args.imitation):
        modes[("imit", d["mode"])] += 1
        rows.append(d)
    for d in read(args.contrastive):
        d = retag(d)
        modes[("contra", d["mode"])] += 1
        rows.append(d)
    if args.shuffle:
        random.Random(args.seed).shuffle(rows)
    op = gzip.open if args.out.endswith(".gz") else open
    with op(args.out, "wt") as w:
        for d in rows:
            w.write(json.dumps(d) + "\n")
    print(f"wrote {len(rows)} samples -> {args.out}")
    for k in sorted(modes):
        print(f"  {k}: {modes[k]}")


if __name__ == "__main__":
    main()
