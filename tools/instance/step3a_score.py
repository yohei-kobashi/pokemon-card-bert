#!/usr/bin/env python3
"""Step 3a: does the teacher's DISTRIBUTION carry value information a hard label does not?

A hard label transmits one thing: which move the labeller picked.  Everything distillation adds
over it lives in the ORDER OF THE LOSERS.  So the question splits in two:

  V1  E[ Q(model's 1st) - mean Q(the others) ]      is the top pick good?
  V2  E[ Q(model's 2nd) - mean Q(the rest) ]        is the tail ordered, or noise below the top?
  CON pairwise concordance with Q                   chance = 50%

Every term is a mean, never a max, so all three are 0 (or 50%) under the null -- the same
unbiased construction rank_probe uses, which is why its +0.0469 +/- 0.0032 for the policy is
directly comparable.  (An earlier read of 72.4% "argmax agreement" was uninterpretable because
max() over noisy Q is upward biased.)

DECISION RULE, fixed before looking:
  V2 not distinguishable from 0  -> the soft tail is noise; hard labels lose nothing; the 9B is
                                    worthless as a distillation teacher regardless of its
                                    playing strength.
  V1 <= policy + 2SE             -> the teacher does not pick better moves than the student
                                    already does; nothing to teach at the top either.

GUARD: legal_mass.  The teacher was trained on engine_v2 self-play prompts; these come from RL
rollouts.  If the format drifted, the model is off-distribution and every number here is noise.
It measured 0.9999 on its own held-out set, so anything much below that invalidates the run.
"""
import argparse
import gzip
import json
import math
import sys
import time

sys.path.insert(0, "/root")
from eval_teacher import n_options, score_decision      # noqa: E402


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
    ap.add_argument("--model", default="unsloth/Qwen3.5-9B-Base")
    ap.add_argument("--adapter", default="/root/out/teacher9b")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--maxlen", type=int, default=1024)
    ap.add_argument("--dump", default=None)
    a = ap.parse_args()

    pts = [json.loads(l) for l in gzip.open(a.points, "rt")]
    if a.limit:
        pts = pts[:a.limit]

    t0 = time.time()
    from unsloth import FastLanguageModel                # noqa: E402
    import torch                                         # noqa: E402
    model, tok = FastLanguageModel.from_pretrained(
        model_name=a.model, max_seq_length=a.maxlen,
        load_in_4bit=False, load_in_16bit=True, full_finetuning=False)
    if a.adapter:
        from peft import PeftModel                       # noqa: E402
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()
    tk = getattr(tok, "tokenizer", tok)
    if tk.pad_token is None:
        tk.pad_token = tk.eos_token
    print("[load] %.1fs  points %d" % (time.time() - t0, len(pts)), flush=True)

    v1, v2, ceil, masses = [], [], [], []
    con_ok = con_n = 0
    skipped = 0
    dump = gzip.open(a.dump, "wt") if a.dump else None
    for r in pts:
        n = len(r["cands"])
        if n_options(r["prompt"]) != n:      # Step 0 said this never happens; assert it anyway
            skipped += 1
            continue
        lp, mass, _ = score_decision(model, tk, torch, r["prompt"], n, a.maxlen)
        idx, q = r["idx"], r["q"]
        s = [lp[i] for i in idx]
        masses.append(mass)

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
            dump.write(json.dumps({"logp": [round(x, 5) for x in lp], "idx": idx, "q": q,
                                   "legal_mass": round(mass, 6)}) + "\n")
        if len(v1) % 500 == 0:
            m, se, _ = mean_se(v1)
            print("  %d/%d  V1 %+.4f+-%.4f  mass %.4f  %.0fs"
                  % (len(v1), len(pts), m, se, sum(masses) / len(masses), time.time() - t0),
                  flush=True)
    if dump:
        dump.close()

    m1, se1, n1 = mean_se(v1)
    m2, se2, n2 = mean_se(v2)
    mc, sec, _ = mean_se(ceil)
    mm = sum(masses) / max(1, len(masses))
    print("\n=== Step 3a: teacher ranking vs playout Q ===", flush=True)
    print("  legal_mass      %.4f   (0.9999 on its own held-out; far below = off-distribution)"
          % mm, flush=True)
    print("  decisions       %d  (skipped %d)" % (n1, skipped), flush=True)
    print("  V1  top pick    %+.4f +- %.4f   (t=%.1f)" % (m1, se1, m1 / max(1e-9, se1)), flush=True)
    print("  V2  2nd pick    %+.4f +- %.4f   (t=%.1f)  n=%d"
          % (m2, se2, m2 / max(1e-9, se2), n2), flush=True)
    print("  concordance     %.2f%%  of %d ranked pairs   (chance 50%%)"
          % (100.0 * con_ok / max(1, con_n), con_n), flush=True)
    print("  ceiling (V1)    %+.4f +- %.4f   perfect ranking of these same decisions"
          % (mc, sec), flush=True)
    print("  policy baseline +0.0469 +- 0.0032  (rank_probe, same metric, other rollout set)",
          flush=True)
    print("[total] %.1f min" % ((time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
