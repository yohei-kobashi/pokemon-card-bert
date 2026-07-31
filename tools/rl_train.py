"""One GRPO update pass from a rollout file (docs/rl_design.md §3-5).

Policy: pi(a|s) = softmax over LEGAL candidates of the CROSS-ENCODER's scalar score for the
(state, candidate) pair. A decision is a listwise choice, and the gradient reaches the model
through the chosen candidate's score relative to the others -- the same parameterisation
train_rerank.py fits with softmax-CE, so RL reweights the SFT objective instead of replacing
it. (Before 2026-07-28 this was a decoder's length-normalised token logprob through a LoRA.)

Advantage (VALUE-FREE by default):
  RAE  — subtract the mean reward of the SAME matchup group (role = pilot x opponent).
         A_game = R_game - mean_{same matchup}(R).            (SPIRAL role-conditioned baseline)
  MARS — terminal reward only (+1/-1), so per-turn cumulative return == terminal; every
         decision in a game inherits A_game, then advantages are whitened across the batch
         (turn-level normalization / variance control).
  (optional) -V(s): NOT used by default; the design forbids value-in-reward and GAE.

  DECISION-LEVEL (added 2026-07-29, --branch-weight). The two rules above give every decision
  in a game the SAME advantage, and that is measured to be the reason RL stalls: 8,064 games
  moved the gate 0.0pt (rl-stage-a-plateau-diagnosis). Where rl_rollout re-played a position
  with each of the top candidates, the record carries `qvals` and we add an ALL-ACTION term,
  sum_k pi_k (Q_k - V) grad log pi_k, which credits that decision alone and also reaches the
  candidates the policy did not play. Still value-free: Q comes from playouts, not a learned
  critic. --branch-weight 0 reproduces the game-level-only update exactly.

Objective (GRPO, clipped):
  ratio  = exp( logpi_new(chosen) - logpi_old(chosen) )      # both at temp=1 from scores
  L      = -mean( min(ratio*A, clip(ratio,1-e,1+e)*A) ) + kl_coef * KL(pi_new || pi_old)
  format penalty is unnecessary here (we only ever score LEGAL candidates).

FULL fine-tune of the 149M reranker (no LoRA, no 4-bit): --model is the policy checkpoint
(the SFT reranker on round 0, the previous RL round after), --out the updated one.

The prompt format lives in rl_config.PROMPT_FMT and must match that checkpoint. rl_rollout
logs prompts built with it and this file re-scores those very strings, so a mismatch does not
just score badly -- it puts the gradient on inputs the policy never plays on.
"""
import argparse
import gzip
import json
import math
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_rollout(path):
    recs = [json.loads(l) for l in gzip.open(path, "rt")]
    rewards = json.load(open(path + ".rewards.json"))
    # map decisions -> games (records are appended per game, in reward order)
    games, i = [], 0
    for gr in rewards:
        n = gr["n_decisions"]
        games.append(dict(matchup=gr["matchup"], reward=gr["reward"],
                          decisions=recs[i:i + n]))
        i += n
    return games


def _pilot_of(matchup):
    return matchup.split("__vs__")[0]


def _advantages(games, boost=None):
    """RAE: A_game = reward - matchup mean. Returns (flat_dec, flat_adv, flat_w) aligned to a
    flat decision list; advantages whitened (MARS turn-level normalization / variance control).

    `boost` = dict(pilots=set_of_low_winrate_pilots, factor=f): the WIN-OVERSAMPLING booster
    (Stage A). For a WINNING game piloted by a low-winrate deck, its decisions get loss weight
    `factor` (else 1.0). This is applied as a LOSS WEIGHT, NOT by replicating games, so the RAE
    per-matchup baseline stays unbiased (replicating only-wins would inflate the matchup mean
    and shrink the very advantage we want). It lifts the density of positive-play signal for
    decks too weak to generate enough wins on their own -- the contrastive learning that a
    ~15%-winrate deck otherwise never gets."""
    by_matchup = defaultdict(list)
    for g in games:
        by_matchup[g["matchup"]].append(g["reward"])
    base = {m: sum(v) / len(v) for m, v in by_matchup.items()}
    bp = (boost or {}).get("pilots", set())
    bf = float((boost or {}).get("factor", 1.0))
    flat_dec, flat_adv, flat_w = [], [], []
    for g in games:
        a = g["reward"] - base[g["matchup"]]
        w = bf if (g["reward"] > 0 and _pilot_of(g["matchup"]) in bp) else 1.0
        for d in g["decisions"]:
            flat_dec.append(d)
            flat_adv.append(a)
            flat_w.append(w)
    if flat_adv:
        mu = sum(flat_adv) / len(flat_adv)
        var = sum((x - mu) ** 2 for x in flat_adv) / max(1, len(flat_adv))
        sd = math.sqrt(var) + 1e-8
        flat_adv = [(x - mu) / sd for x in flat_adv]
    return flat_dec, flat_adv, flat_w


_ACT = "[ACT]\n"


def _logpi_group(model, tok, torch, device, maxlen, decisions, grad):
    """log pi(chosen|s) and the full pi, for a GROUP of decisions, in ONE forward.

    The per-decision version issued one forward per decision -- about six (state, candidate)
    pairs of ~300 tokens, which leaves a 4090 almost idle. Measured on the first real round:
    1,413 decisions took 132 s (10.7/s), so a 768-game round (~45k decisions) would have
    spent ~70 min in the update against ~21 min in the rollout. train_rerank.py already
    solves this -- gather until ~pair_batch pairs, tokenize and forward them together, then
    split the logits back per decision -- and the listwise softmax is per decision either
    way, so batching changes throughput and nothing else.

    Returns [(logpi_chosen, pi_row)] aligned to `decisions`.
    """
    pairs, owner = [], []
    for di, d in enumerate(decisions):
        state = d["prompt"][len(_ACT):] if d["prompt"].startswith(_ACT) else d["prompt"]
        for c in d["cands"]:
            pairs.append([state, c])
            owner.append(di)
    enc = tok(pairs, padding=True, truncation="only_first", max_length=maxlen,
              return_tensors="pt").to(device)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        flat = model(**enc).logits.squeeze(-1).float()
        out = []
        pos = 0
        for d in decisions:
            n = len(d["cands"])
            pi = torch.log_softmax(flat[pos:pos + n], 0)
            out.append((pi[d["chosen"]], pi))
            pos += n
    return out


def _branch_advantages(dec):
    """All-action advantages for the decisions rl_rollout branched (docs: qvals).

    The game-level term above has to spread one scalar over ~70 decisions, which is the
    measured cause of the plateau. Where the rollout re-played the position with each of the
    top candidates we can do better: at THIS state candidate k is worth Q_k, so the gradient
    of E_{a~pi}[Q] is  sum_k pi_k (Q_k - V) grad log pi_k  with V = sum_k pi_k Q_k over the
    branched candidates. Every branched candidate contributes -- including the ones the
    policy did NOT play, which the sampled-action-only surrogate can never reach.

    Returns a list aligned to `dec`: None, or [indices, advantages, policy weights].
    Advantages are SCALED (not centred) to the game-level term's units: sum_k pi_k A_k = 0
    holds per decision by construction and centring would destroy it.
    """
    out = [None] * len(dec)
    pool = []
    for j, d in enumerate(dec):
        q = d.get("qvals")
        s = d.get("scores") or []
        if not q or len(q) != len(s):
            continue
        idx = [i for i, v in enumerate(q) if v is not None]
        if len(idx) < 2:
            continue
        m = max(s[i] for i in idx)
        ex = [math.exp(s[i] - m) for i in idx]
        Z = sum(ex) or 1.0
        p = [e / Z for e in ex]
        V = sum(pi * q[i] for pi, i in zip(p, idx))
        a = [q[i] - V for i in idx]
        out[j] = [idx, a, p]
        pool.extend(a)
    if pool:
        var = sum(x * x for x in pool) / len(pool)      # mean is 0 by construction
        sd = math.sqrt(var) + 1e-8
        for t in out:
            if t is not None:
                t[1] = [x / sd for x in t[1]]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rollout", required=True, help="rollout .jsonl.gz from rl_rollout.py")
    ap.add_argument("--model", required=True,
                    help="policy checkpoint dir (the SFT reranker on round 0, the "
                         "previous RL round after). Full weights, not an adapter.")
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="gradient checkpointing (slower, less memory)")
    ap.add_argument("--out", required=True, help="save the updated policy dir")
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--kl-coef", type=float, default=0.02)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--maxlen", type=int, default=1024)
    ap.add_argument("--decision-frac", type=float, default=1.0,
                    help="fraction of decisions to keep for the UPDATE (rollout is unchanged). "
                         "Terminal-only reward means every decision in a game carries the SAME "
                         "advantage, so a uniform sample over the flat decision list is an "
                         "unbiased estimate of the same mean gradient -- and decisions within a "
                         "game are correlated, so the variance cost is below sqrt(1/frac). "
                         "NOT free: each round gets noisier, and the honest comparison is "
                         "'more rounds per hour', not 'same round, less time'.")
    ap.add_argument("--decision-seed", type=int, default=1234,
                    help="seed for --decision-frac; vary it to draw a different sample")
    ap.add_argument("--pair-batch", type=int, default=256,
                    help="(state, candidate) pairs per forward -- the GPU unit")
    ap.add_argument("--minibatch", type=int, default=4,
                    help="GROUPS per optimizer step -- the optimizer unit")
    ap.add_argument("--win-boost", action="store_true",
                    help="Stage A: up-weight the loss of WINNING decisions from low-winrate "
                         "pilots (< --win-boost-thresh) by --win-boost-factor, to lift the "
                         "positive-play signal density for decks too weak to self-generate wins.")
    ap.add_argument("--branch-weight", type=float, default=1.0,
                    help="scale of the decision-level (all-action) term relative to the "
                         "game-level GRPO term, for decisions rl_rollout branched. 0 disables "
                         "it and reproduces the game-level-only update that plateaued.")
    ap.add_argument("--win-boost-thresh", type=float, default=0.35)
    ap.add_argument("--win-boost-factor", type=float, default=3.0)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # CROSS-ENCODER, FULL fine-tune. The decoder path (4-bit QLoRA + PeftModel + set_adapter)
    # is gone with the decoder: a 149M reranker fits on the 4090 in bf16 with room to spare,
    # and LoRA existed mainly to keep two policies swappable on one base -- which Stage A no
    # longer needs now that the opponent is 100% engine_v2 (no second model at all).
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.truncation_side = "left"          # overflow drops the HEAD, never the board+menu
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16).to("cuda")
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
    model.train()
    device = next(model.parameters()).device
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    games = _load_rollout(args.rollout)
    boost = None
    if args.win_boost:
        wins = defaultdict(int); tot = defaultdict(int)
        for g in games:
            p = _pilot_of(g["matchup"]); tot[p] += 1; wins[p] += 1 if g["reward"] > 0 else 0
        low = {p for p in tot if wins[p] / max(1, tot[p]) < args.win_boost_thresh}
        boost = dict(pilots=low, factor=args.win_boost_factor)
        print(f"rl_train: win-boost x{args.win_boost_factor} on {len(low)} low-winrate pilots "
              f"(<{args.win_boost_thresh:.0%}): {sorted(low)[:8]}{'...' if len(low)>8 else ''}",
              flush=True)
    dec, adv, wts = _advantages(games, boost=boost)
    if args.decision_frac < 1.0:
        # Subsample AFTER _advantages: the whitening statistics (mu, sd) must come from every
        # decision, or the advantage scale itself shifts with the sampling rate. Uniform over
        # the FLAT list, not a fixed count per game -- long games legitimately contribute more
        # gradient (the REINFORCE sum over a trajectory), and a per-game quota would silently
        # reweight them.
        import random as _rs
        # Branched decisions are NEVER dropped. They cost real playout time and, unlike the
        # rest, their advantage is attributed to the decision itself -- subsampling them would
        # throw away the only signal this round bought that the game scalar cannot provide.
        forced = [i for i, d in enumerate(dec) if d.get("qvals")]
        rest = [i for i, d in enumerate(dec) if not d.get("qvals")]
        want = max(1, int(len(dec) * args.decision_frac)) - len(forced)
        keep = forced + (_rs.Random(args.decision_seed).sample(rest, want)
                         if 0 < want < len(rest) else (rest if want >= len(rest) else []))
        keep.sort()
        dec = [dec[i] for i in keep]
        adv = [adv[i] for i in keep]
        wts = [wts[i] for i in keep]
        print(f"decision-frac {args.decision_frac}: {len(dec)} decisions kept "
              f"({len(forced)} branched, always kept)", flush=True)
    # after subsampling: the branch list must stay aligned to `dec`
    badv = _branch_advantages(dec) if args.branch_weight > 0 else [None] * len(dec)
    n_br = sum(1 for t in badv if t is not None)
    if n_br:
        print(f"rl_train: decision-level term on {n_br} branched decisions "
              f"({100.0*n_br/max(1,len(dec)):.1f}% of the update), weight {args.branch_weight}",
              flush=True)
    print(f"rl_train: {len(games)} games, {len(dec)} decisions, "
          f"pilot winrate {sum(1 for g in games if g['reward']>0)/max(1,len(games)):.1%}", flush=True)

    for ep in range(args.epochs):
        order = list(range(len(dec)))
        import random as _r; _r.Random(ep).shuffle(order)
        # group decisions until ~pair_batch (state, candidate) pairs -- the unit the GPU
        # actually likes. minibatch stays the OPTIMIZER's unit: groups per opt.step().
        groups, cur, npairs = [], [], 0
        for k in order:
            cur.append(k); npairs += len(dec[k]["cands"])
            if npairs >= args.pair_batch:
                groups.append(cur); cur, npairs = [], 0
        if cur:
            groups.append(cur)
        step_loss = 0.0; step_n = 0
        opt.zero_grad()
        for gi, grp in enumerate(groups):
            ds = [dec[k] for k in grp]
            res = _logpi_group(model, tok, torch, device, args.maxlen, ds, grad=True)
            loss = 0.0
            for (new_lp, new_pi), k, d in zip(res, grp, ds):
                A = adv[k]; W = wts[k]
                old_pi = torch.log_softmax(torch.tensor(d["scores"]), 0).to(new_pi.device)
                old_lp = float(old_pi[d["chosen"]])
                ratio = torch.exp(new_lp - old_lp)
                pg = -torch.min(ratio * A, torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * A)
                kl = (new_pi.exp() * (new_pi - old_pi)).sum()
                loss = loss + W * (pg + args.kl_coef * kl)
                bt = badv[k]
                if bt is not None:
                    # all-action term: no importance ratio and no clipping. These Q come from
                    # counterfactual playouts, not from the behaviour policy's samples, so
                    # there is no off-policy correction to make -- clipping them would only
                    # throttle the one signal that is actually attributed to this decision.
                    idx, aa, pp = bt
                    pgb = -sum(w * a * new_pi[i] for i, a, w in zip(idx, aa, pp))
                    loss = loss + W * args.branch_weight * pgb
                step_loss += float(pg); step_n += 1
            (loss / max(1, len(ds)) / args.minibatch).backward()
            if (gi + 1) % args.minibatch == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad()
        if len(groups) % args.minibatch:            # flush the tail
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad()
        print(f"  epoch {ep}: {len(groups)} groups, mean pg-loss "
              f"{step_loss/max(1,step_n):+.4f}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)          # the domain tokens live in the tokenizer
    print(f"saved policy -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
