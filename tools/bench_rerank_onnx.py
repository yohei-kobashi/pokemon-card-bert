"""Benchmark the INT8 ONNX reranker on CPU under Kaggle-like constraints.

The competition runtime gives 4 vCPU (AMD EPYC) and a 600 s CUMULATIVE thinking bank per
game, so the question is: (per-decision latency) x (scored decisions per game) < 600 s?

A cross-encoder re-encodes the FULL ~830-token state once PER CANDIDATE (no KV reuse), so
cost scales with candidates x state length -- this is the reranker's main deploy risk.
Decisions are replayed from the real training records (state + candidate texts), so the
candidate counts and sequence lengths are the true deploy distribution. Only onnxruntime +
tokenizers are used (torch/transformers are NOT in the submission bundle).

Usage:
  python tools/bench_rerank_onnx.py --onnx /root/onnx/rerank/model_int8.onnx \
      --tokenizer /root/out/rerank_gte_mp --data <rerank.jsonl.gz> --n 120 --threads 4
"""
import argparse
import gzip
import json
import os
import statistics
import sys
import time

# Scored decisions per game for ONE side, MEASURED by instrumenting real games
# (eval_rerank.py reports mean_scored_decisions): 59-90 across matchups.
#
# Do NOT derive this from the training-record count (1,523,895 / 23,436 games = 65). That
# undercounts: build_rerank keeps WINNER-side decisions only and _emit drops any decision
# whose candidate texts collapse to <2 after dedup, while the live agent must score every
# real choice on its own side, win or lose. Using 65 understated a game by ~25%.
DECISIONS_PER_GAME = 80.0
GAME_BUDGET_S = 600.0


def _load_records(data, n, stride=977):
    recs = []
    with gzip.open(data, "rt") as fh:
        for i, line in enumerate(fh):
            if i % stride == 0:
                recs.append(json.loads(line))
                if len(recs) >= n:
                    break
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--tokenizer", required=True, help="dir with tokenizer.json")
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--remap", default="", help="vocab_remap.npy for a vocab-pruned --onnx; "
                    "the remap is a real per-decision cost so it is timed with the rest")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = str(args.threads)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # Time the SHIPPED scorer, not a re-implementation of it. A private copy of the
    # tokenizer/session setup here drifts from lm/rerank_scorer.py silently (it already
    # lacked the deploy scorer's direction="left" truncation), and then the number that
    # decides whether we submit describes code that is not in the tarball.
    from lm.rerank_scorer import OnnxRerankerScorer          # noqa: E402

    scorer = OnnxRerankerScorer(args.onnx, args.tokenizer, max_len=args.max_len,
                                threads=args.threads, remap=args.remap or None,
                                time_budget=float("inf"))

    recs = _load_records(args.data, args.n + args.warmup)
    lat, n_cands, seqs = [], [], []
    for k, r in enumerate(recs):
        t0 = time.perf_counter()
        scorer.score(r["state"], r["candidates"])
        t2 = time.perf_counter()
        if k >= args.warmup:
            lat.append(t2 - t0)
            n_cands.append(len(r["candidates"]))
            seqs.append(len(scorer.tok.encode(r["state"], r["candidates"][0]).ids))

    lat_s = sorted(lat)
    res = dict(
        n_decisions=len(lat), threads=args.threads,
        mean_candidates=statistics.mean(n_cands), mean_seq=statistics.mean(seqs),
        p90_seq=lat_s and seqs[int(0.9 * (len(seqs) - 1))],
        mean_latency_s=statistics.mean(lat), median_latency_s=statistics.median(lat),
        p90_latency_s=lat_s[int(0.9 * (len(lat_s) - 1))], max_latency_s=lat_s[-1],
        per_pair_ms=1000 * sum(lat) / sum(n_cands),
        decisions_per_game=DECISIONS_PER_GAME)
    res["projected_game_s"] = res["mean_latency_s"] * DECISIONS_PER_GAME
    res["budget_use_pct"] = 100 * res["projected_game_s"] / GAME_BUDGET_S
    res["verdict"] = "FITS" if res["projected_game_s"] < GAME_BUDGET_S else "OVER BUDGET"

    print(f"threads={args.threads}  decisions={res['n_decisions']} "
          f"(through lm/rerank_scorer.OnnxRerankerScorer = the shipped code)")
    print(f"  candidates/decision  mean {res['mean_candidates']:.2f}")
    print(f"  seq len              mean {res['mean_seq']:.0f}")
    print(f"  latency/decision     mean {res['mean_latency_s']:.3f}s  "
          f"median {res['median_latency_s']:.3f}s  p90 {res['p90_latency_s']:.3f}s  "
          f"max {res['max_latency_s']:.3f}s")
    print(f"  per (state,cand) pair {res['per_pair_ms']:.1f} ms")
    print(f"  PROJECTED PER GAME   {res['projected_game_s']:.0f}s "
          f"({res['budget_use_pct']:.0f}% of the {GAME_BUDGET_S:.0f}s bank)  -> {res['verdict']}")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
