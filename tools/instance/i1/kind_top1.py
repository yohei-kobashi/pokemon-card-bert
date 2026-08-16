"""top1 broken down by the KIND of decision, on ONE held-out set, for several checkpoints.

Play-time probing showed the LM's residual error concentrates on energy attachment: deferring
`attach` decisions to engine_v2 recovers +11.4pt of v36's alakazam_nz deficit while deferring
`retreat` recovers nothing. That was measured on states the LM itself reached, so it cannot
separate two causes:
  (a) the model is simply WORSE at attach decisions (a learning problem, visible offline), or
  (b) it is fine offline and only fails on the off-distribution states its own play produces.
Held-out top1 per kind answers that: it is measured on ENGINE-piloted states, so any gap here
is (a). If attach top1 matches the other kinds, the defect is (b) and no amount of reweighting
the training mix will fix it.

The kind is taken from the LABELLED candidate (what engine_v2 chose), so the buckets are the
same rows for every checkpoint.
"""
import argparse
import collections
import json
import os
import sys

ROOT = "/root/ptcg/repo"
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="comma-separated checkpoint dirs")
    ap.add_argument("--cache", required=True, help="eval split json (list of rows)")
    ap.add_argument("--n", type=int, default=0, help="cap rows (0 = all)")
    ap.add_argument("--pair-batch", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=640)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    rows = json.load(open(args.cache))
    if args.n:
        rows = rows[:args.n]
    kinds = [r["candidates"][r["chosen"]].split(":")[0].split("@")[0] for r in rows]
    tally = collections.Counter(kinds)
    print("eval rows %d; kinds: %s" % (len(rows), ", ".join(
        "%s %d" % (k, v) for k, v in tally.most_common(9))), flush=True)

    out = {}
    for mdir in args.models.split(","):
        tok = AutoTokenizer.from_pretrained(mdir)
        tok.truncation_side = "left"
        model = AutoModelForSequenceClassification.from_pretrained(
            mdir, trust_remote_code=True, dtype=torch.bfloat16).to("cuda").eval()
        hit = collections.Counter()
        tot = collections.Counter()
        i = 0
        while i < len(rows):
            grp, npairs = [], 0
            while i < len(rows) and npairs < args.pair_batch:
                grp.append(rows[i]); npairs += len(rows[i]["candidates"]); i += 1
            pairs, owner = [], []
            for ri, r in enumerate(grp):
                for c in r["candidates"]:
                    pairs.append([r["state"], c]); owner.append(ri)
            enc = tok(pairs, padding=True, truncation="only_first",
                      max_length=args.max_len, return_tensors="pt").to("cuda")
            with torch.no_grad():
                logits = model(**enc).logits.squeeze(-1).float().tolist()
            per = [[] for _ in grp]
            for k, ri in enumerate(owner):
                per[ri].append(logits[k])
            for ri, r in enumerate(grp):
                s = per[ri]
                pred = max(range(len(s)), key=lambda j: s[j])
                kd = r["candidates"][r["chosen"]].split(":")[0].split("@")[0]
                tot[kd] += 1
                hit[kd] += (pred == r["chosen"])
        out[os.path.basename(mdir)] = (hit, tot)
        del model
        torch.cuda.empty_cache()
        print("  scored %s" % os.path.basename(mdir), flush=True)

    names = list(out)
    ks = [k for k, _ in tally.most_common() if tally[k] >= 30]
    print()
    print("%-14s %7s" % ("kind", "n") + "".join("%10s" % n[-4:] for n in names)
          + "   chance")
    for k in ks:
        h0, t0 = out[names[0]]
        line = "%-14s %7d" % (k, t0[k])
        for n in names:
            h, t = out[n]
            line += "%9.1f%%" % (100.0 * h[k] / max(1, t[k]))
        print(line)
    print("%-14s %7d" % ("ALL", sum(out[names[0]][1].values()))
          + "".join("%9.1f%%" % (100.0 * sum(out[n][0].values()) / max(1, sum(out[n][1].values())))
                    for n in names))


if __name__ == "__main__":
    main()
