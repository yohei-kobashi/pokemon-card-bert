#!/usr/bin/env python3
"""Policy gradient on mirror-paired rollouts, with the shaped return of docs/rl_dusknoir_design.md.

No engine_v2, no value net. The baseline is the GROUP: trajectories that share a seed share the
deal, because mirror mode fixes the shuffle for both seats, so subtracting the group mean
removes draw luck rather than estimating it away ([[mirror-shuffle-mode]]: the null is exactly
0). `rl-design-value-free` is the constraint this satisfies.

    return    R_t = 3*outcome*gamma^(T-t) + sum_{u>=t} gamma^(u-t) * [gamma*Phi_{u+1} - Phi_u]
    advantage A_t = R_t - mean over the group at the SAME decision index
    loss      -A_t * log pi(a_t|s_t) + beta_kl * KL(pi || pi_ref) - beta_h * H(pi)

Per-decision credit is the point. `rl-stage-a-plateau-diagnosis` measured that one win/loss bit
spread uniformly over ~80 decisions is flat for twelve rounds; the potential is what makes two
decisions in one game score differently.

    PYTHONPATH=cg-lib python3 tools/rl_pg_train.py --data /root/rl/roll_1.jsonl.gz \\
        --model /root/out/d41_r8 --out /root/out/rl_r1
"""

import argparse
import collections
import gzip
import json
import math
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

def _seed_means(pairs):
    agg = collections.defaultdict(list)
    for s, m in pairs:
        agg[s].append(m)
    return [(s, st.mean(v)) for s, v in agg.items()]


W_TERMINAL = 3.0            # a win must outweigh any Phi swing; Phi spans roughly [-6, +8]


def load(path, gamma):
    """Rollout file -> per-decision samples carrying the shaped return-to-go."""
    groups = collections.defaultdict(list)
    with gzip.open(path, "rt") as f:
        for line in f:
            d = json.loads(line)
            if d.get("header"):
                continue
            groups[d["seed"]].append(d)
    out, stats = [], {"traj": 0, "dec": 0, "fallback": 0}
    per_traj_mean, within_sd = [], []
    for seed, traj in groups.items():
        per_traj = []
        for tr in traj:
            dec = tr["decisions"]
            stats["traj"] += 1
            stats["fallback"] += tr.get("fallback", 0)
            res = tr.get("result")
            # Per SEAT: the outcome of a decision is the outcome for the player who made it.
            rets = [0.0] * len(dec)
            for seat in (0, 1):
                idx = [i for i, x in enumerate(dec) if x["seat"] == seat]
                if not idx:
                    continue
                won = 0.0 if res not in (0, 1) else (1.0 if res == seat else -1.0)
                acc = W_TERMINAL * won
                # walk backwards: return-to-go, shaping added as a difference of potentials
                for j in range(len(idx) - 1, -1, -1):
                    i = idx[j]
                    nxt = dec[idx[j + 1]]["phi"] if j + 1 < len(idx) else dec[i]["phi"]
                    acc = gamma * acc + (gamma * nxt - dec[i]["phi"])
                    rets[i] = acc
            per_traj.append((tr, rets))
            stats["dec"] += len(dec)
            if rets:
                per_traj_mean.append((seed, st.mean(rets)))
        # STEP 4 OF THE LADDER: same-seed spread must be smaller than across-seed spread, or
        # the group baseline is buying nothing and a plain mean would do as well.
        if len(per_traj) > 1:
            gm = [st.mean(r) for _, r in per_traj if r]
            if len(gm) > 1:
                within_sd.append(st.pstdev(gm))     # spread WITHIN one deal
        # advantage: group mean at the same decision INDEX, so like is compared with like
        n = min(len(r) for _, r in per_traj) if per_traj else 0
        for t in range(n):
            vals = [r[t] for _, r in per_traj]
            mu = st.mean(vals)
            for (tr, r) in per_traj:
                d = tr["decisions"][t]
                a = r[t] - mu
                if abs(a) < 1e-9:
                    continue
                out.append({"prompt": d["prompt"], "cands": d["cands"],
                            "chosen": d["chosen"], "adv": a})
    if within_sd and len({s for s, _ in per_traj_mean}) > 1:
        # The claim the design makes: fixing the deal removes most of the variance, so the
        # group mean is a baseline rather than an estimate. Compare the spread of trajectory
        # returns WITHIN a deal against the spread of the deals' own means.
        w = st.mean(within_sd)
        b = st.pstdev([m for _, m in _seed_means(per_traj_mean)])
        print("[pairing] within-deal sd %.3f | between-deal sd %.3f | pairing removes %.0f%% "
              "of the variance" % (w, b, 100 * (b ** 2) / max(1e-9, b ** 2 + w ** 2)),
              flush=True)
    print("[data] %d trajectories, %d decisions, %d usable samples | fallback decisions %d"
          % (stats["traj"], stats["dec"], len(out), stats["fallback"]), flush=True)
    if stats["fallback"] > 0.02 * max(1, stats["dec"]):
        sys.exit("more than 2%% of decisions were answered by the engine_v2 fallback -- the "
                 "rollout is not on-policy; fix that before training on it")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True, help="checkpoint to continue AND the reference")
    ap.add_argument("--ref", default="", help="defaults to --model; keep it PINNED across "
                                              "rounds or beta stops measuring total drift")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=8, help="decisions per step")
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--maxlen", type=int, default=512)
    ap.add_argument("--beta-kl", type=float, default=0.05)
    ap.add_argument("--beta-h", type=float, default=0.01)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--probe", action="store_true",
                    help="overfit 200 samples with no KL: a trainer that cannot collapse THIS "
                         "must not spend a round")
    a = ap.parse_args()

    rows = load(a.data, a.gamma)
    if not rows:
        sys.exit("no usable samples (every advantage was zero -- did the group diverge?)")
    if a.probe:
        rows, a.epochs, a.beta_kl, a.lr = rows[:200], 12.0, 0.0, 3e-5

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.truncation_side = "left"          # the menu is last; a right cut deletes the options
    model = AutoModelForSequenceClassification.from_pretrained(a.model).to(dev)
    ref = AutoModelForSequenceClassification.from_pretrained(a.ref or a.model).to(dev).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)

    def logits_for(r, m):
        enc = tok([r["prompt"]] * len(r["cands"]), r["cands"], return_tensors="pt",
                  padding=True, truncation=True, max_length=a.maxlen).to(dev)
        return m(**enc).logits.squeeze(-1)

    import random
    random.Random(0).shuffle(rows)
    n_steps = int(len(rows) * a.epochs)
    losses, ents, kls = [], [], []
    step = 0
    for i in range(n_steps):
        r = rows[i % len(rows)]
        if len(r["cands"]) < 2 or not (0 <= r["chosen"] < len(r["cands"])):
            continue
        s = logits_for(r, model) / a.temp
        logp = torch.log_softmax(s, dim=-1)
        pol = -r["adv"] * logp[r["chosen"]]
        ent = -(logp.exp() * logp).sum()
        with torch.no_grad():
            sr = logits_for(r, ref) / a.temp
            logq = torch.log_softmax(sr, dim=-1)
        kl = (logp.exp() * (logp - logq)).sum()
        loss = (pol + a.beta_kl * kl - a.beta_h * ent) / a.accum
        loss.backward()
        losses.append(float(pol)); ents.append(float(ent)); kls.append(float(kl))
        step += 1
        if step % a.accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad()
        if step % 200 == 0:
            print("  step %d  pg %.4f  H %.3f  KL %.4f"
                  % (step, st.mean(losses[-200:]), st.mean(ents[-200:]),
                     st.mean(kls[-200:])), flush=True)
    print("FINAL pg %.4f  H %.3f  KL %.4f  over %d steps"
          % (st.mean(losses or [0]), st.mean(ents or [0]), st.mean(kls or [0]), step))
    if a.probe:
        first, last = st.mean(losses[:100] or [0]), st.mean(losses[-100:] or [0])
        print("PROBE %s: %.4f -> %.4f" % ("OK" if last < first - 0.05 else "FAILED",
                                          first, last))
        return
    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print("[saved] %s" % a.out)


if __name__ == "__main__":
    main()
