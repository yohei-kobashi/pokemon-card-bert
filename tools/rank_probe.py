"""Can the cross-encoder learn to rank moves by their playout value AT ALL?

This separates two very different explanations for the RL plateau, which no gate can tell
apart:

  (a) OPTIMISATION limit -- the representation could rank these moves, but the RL gradient
      (one scalar per game, or a 16%-weight all-action term) never finds it. Then RL is
      fixable: change the objective, the weight, the branch density.
  (b) REPRESENTATION limit -- the prompt does not carry what separates these moves, so no
      amount of RL can work and the fix is upstream (prompt, features, model size).

The test: take the branch points rl_rollout already recorded (state, candidates, playout Q
for each) and fit the SAME model SUPERVISED to rank candidates by Q. Supervised learning is
the friendliest possible optimiser -- dense, direct, no credit assignment. If held-out
ranking still does not move, (b) is the answer.

METRIC (must be bias-free). Not "how often is the model's pick the argmax of Q": with 2
playouts Q is in {-1,0,+1}, ties are everywhere and max() over noisy estimates is upward
biased (winner's curse), which is why the earlier read of 72.4% was uninterpretable. Instead:

    E[ Q(model's top-scored candidate) - mean Q(the others) ]

Every term is an average, never a max, so it is unbiased under the null. 0 = the ranking
carries no value information. The current policy measures +0.0063 +/- 0.0058 (t=1.1) -- i.e.
chance. Perfect ranking of K=4 with signal sd 0.165 would be about +0.17, so the policy is
capturing ~4% of what is there.

CONTROL: the same fit with Q labels SHUFFLED across branch points. If the shuffled run also
improves, the improvement is leakage or a metric bug, not learning.

Run:  python tools/rank_probe.py --rollouts '/root/out/rlDL/A_r*.jsonl.gz' \
          --model /root/out/rlBIG/A_r4_policy [--epochs 2] [--shuffle-control]
"""
import argparse
import glob
import gzip
import json
import math
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)


def load_branch_points(patterns):
    """Decisions that carry playout values, as (prompt, cands, idx, q)."""
    out = []
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            for line in gzip.open(path, "rt"):
                d = json.loads(line)
                q = d.get("qvals")
                if not q:
                    continue
                idx = [i for i, v in enumerate(q) if v is not None]
                if len(idx) < 2:
                    continue
                cands = d.get("cands") or []
                if len(cands) != len(q):
                    continue
                if len(set(q[i] for i in idx)) < 2:
                    continue          # no gradient and no metric signal: every Q identical
                out.append((d["prompt"], cands, idx, [q[i] for i in idx]))
    return out


def metric(model, tok, torch, device, data, maxlen, batch_pairs=256):
    """E[ Q(top-scored) - mean Q(others) ], and its SE. Unbiased under the null."""
    model.eval()
    vals = []
    with torch.no_grad():
        for prompt, cands, idx, qs in data:
            pairs = [[prompt, cands[i]] for i in idx]
            enc = tok(pairs, padding=True, truncation="only_first", max_length=maxlen,
                      return_tensors="pt").to(device)
            s = model(**enc).logits.squeeze(-1).float().tolist()
            if not isinstance(s, list):
                s = [s]
            top = max(range(len(idx)), key=lambda j: s[j])
            others = [qs[j] for j in range(len(idx)) if j != top]
            if others:
                vals.append(qs[top] - sum(others) / len(others))
    n = len(vals)
    m = sum(vals) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in vals) / n)
    return m, sd / math.sqrt(n), n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rollouts", nargs="+", required=True)
    ap.add_argument("--model", required=True, help="checkpoint to start from")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--tau", type=float, default=0.5,
                    help="soft-target temperature: target ∝ exp(Q/tau) over the branched "
                         "candidates. Q is coarse (2 playouts -> {-1,0,+1}), so a soft target "
                         "is the honest label; a hard argmax would train on ties.")
    ap.add_argument("--maxlen", type=int, default=1024)
    ap.add_argument("--holdout", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle-control", action="store_true",
                    help="permute the Q labels across branch points. The fit should then move "
                         "NOTHING on held-out; if it does, the metric or the split leaks.")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    data = load_branch_points(args.rollouts)
    print(f"branch points with a usable Q spread: {len(data)}", flush=True)
    if len(data) < 200:
        raise SystemExit("too few branch points to fit or to measure")
    rng = random.Random(args.seed)
    rng.shuffle(data)
    n_hold = int(len(data) * args.holdout)
    hold, train = data[:n_hold], data[n_hold:]

    if args.shuffle_control:
        qs = [d[3] for d in train]
        rng.shuffle(qs)
        train = [(p, c, i, q) for (p, c, i, _), q in zip(train, qs)]
        print("SHUFFLE CONTROL: training Q labels permuted across branch points", flush=True)
    print(f"train {len(train)}  held-out {len(hold)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.truncation_side = "left"
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16).to("cuda")
    model.gradient_checkpointing_enable()
    device = next(model.parameters()).device

    m0, se0, n0 = metric(model, tok, torch, device, hold, args.maxlen)
    print(f"BEFORE  held-out E[Q(top) - mean Q(others)] = {m0:+.4f} +/- {se0:.4f} "
          f"(n={n0}, t={m0/se0:+.1f})", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    for ep in range(args.epochs):
        model.train()
        rng.shuffle(train)
        tot, nb = 0.0, 0
        for step, (prompt, cands, idx, qs) in enumerate(train):
            pairs = [[prompt, cands[i]] for i in idx]
            enc = tok(pairs, padding=True, truncation="only_first", max_length=args.maxlen,
                      return_tensors="pt").to(device)
            s = model(**enc).logits.squeeze(-1).float()
            logp = torch.log_softmax(s, 0)
            t = torch.tensor(qs, device=device, dtype=torch.float32) / args.tau
            target = torch.softmax(t, 0)
            loss = -(target * logp).sum()
            (loss / 8).backward()
            tot += float(loss); nb += 1
            if (step + 1) % 8 == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad()
        opt.step(); opt.zero_grad()
        m, se, n = metric(model, tok, torch, device, hold, args.maxlen)
        print(f"epoch {ep}: train loss {tot/max(1,nb):.4f} | held-out "
              f"E[Q(top) - mean Q(others)] = {m:+.4f} +/- {se:.4f} (t={m/se:+.1f})", flush=True)

    print()
    print("READ: a held-out move well above 0 means the representation CAN rank these moves "
          "and the RL gradient is the bottleneck. Flat at 0 -- with the shuffle control also "
          "flat -- means the prompt does not carry what separates them, and no RL objective "
          "will fix that.")


if __name__ == "__main__":
    main()
