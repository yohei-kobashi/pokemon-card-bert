#!/usr/bin/env python3
"""The same Step 3a metrics for the cross-encoder student, on the SAME points.

`rerank_gte_v37` is the fair comparator: it was trained on v37 hard labels, and these branch
points are v37 prompts, so neither side is off-distribution.  If the teacher's V1/V2 do not
exceed this, distilling from the teacher cannot beat the hard labels we already use.
"""
import argparse
import gzip
import json
import math
import time


def mean_se(xs):
    n = len(xs)
    if n < 2:
        return 0.0, 0.0, n
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / n)
    return m, sd / math.sqrt(n), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", default="/root/step3a_points.jsonl.gz")
    ap.add_argument("--model", default="/root/out/rerank_gte_v37")
    ap.add_argument("--maxlen", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dump", default=None)
    a = ap.parse_args()

    pts = [json.loads(l) for l in gzip.open(a.points, "rt")]
    if a.limit:
        pts = pts[:a.limit]

    t0 = time.time()
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model).to(dev).eval()
    print("[load] %.1fs  points %d  %s" % (time.time() - t0, len(pts), a.model), flush=True)

    dump = gzip.open(a.dump, "wt") if a.dump else None
    v1, v2, ceil = [], [], []
    con_ok = con_n = 0
    with torch.no_grad():
        for r in pts:
            idx, q = r["idx"], r["q"]
            pairs = [[r["prompt"], r["cands"][i]] for i in idx]
            enc = tok(pairs, padding=True, truncation="only_first", max_length=a.maxlen,
                      return_tensors="pt").to(dev)
            s = model(**enc).logits.squeeze(-1).float().tolist()
            if not isinstance(s, list):
                s = [s]
            order = sorted(range(len(idx)), key=lambda j: -s[j])
            t1 = order[0]
            others = [q[j] for j in range(len(idx)) if j != t1]
            v1.append(q[t1] - sum(others) / len(others))
            ceil.append(max(q) - (sum(q) - max(q)) / (len(q) - 1))
            if len(idx) >= 3:
                t2 = order[1]
                rest = [q[j] for j in range(len(idx)) if j not in (t1, t2)]
                v2.append(q[t2] - sum(rest) / len(rest))
            for i in range(len(idx)):
                for j in range(i + 1, len(idx)):
                    if q[i] == q[j]:
                        continue
                    con_n += 1
                    con_ok += int((s[i] > s[j]) == (q[i] > q[j]))
            if dump:
                dump.write(json.dumps({"s": [round(x,5) for x in s], "idx": idx, "q": q}) + "\n")
            if len(v1) % 2000 == 0:
                m, se, _ = mean_se(v1)
                print("  %d/%d  V1 %+.4f+-%.4f  %.0fs"
                      % (len(v1), len(pts), m, se, time.time() - t0), flush=True)

    if dump:
        dump.close()
    m1, se1, n1 = mean_se(v1)
    m2, se2, n2 = mean_se(v2)
    mc, sec, _ = mean_se(ceil)
    print("\n=== Step 3a: STUDENT (%s) ===" % a.model, flush=True)
    print("  decisions       %d" % n1, flush=True)
    print("  V1  top pick    %+.4f +- %.4f   (t=%.1f)" % (m1, se1, m1 / max(1e-9, se1)), flush=True)
    print("  V2  2nd pick    %+.4f +- %.4f   (t=%.1f)  n=%d"
          % (m2, se2, m2 / max(1e-9, se2), n2), flush=True)
    print("  concordance     %.2f%%  of %d ranked pairs   (chance 50%%)"
          % (100.0 * con_ok / max(1, con_n), con_n), flush=True)
    print("  ceiling (V1)    %+.4f +- %.4f" % (mc, sec), flush=True)
    print("[total] %.1f min" % ((time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
