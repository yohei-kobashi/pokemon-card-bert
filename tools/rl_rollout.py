"""RL rollout generation (docs/rl_design.md §3): the LM policy PLAYS games and records,
at every real decision, the data GRPO needs.

The policy is NOT free generation. Each LEGAL candidate is scored by the cross-encoder (one
scalar per (state, candidate) pair); the policy is a softmax over those candidate scores --
the same parameterisation train_rerank.py fits, so RL reweights the SFT objective.
So a "rollout" logs, per decision:
    prompt, candidate strings, chosen index, old per-candidate scores (logits), matchup key
and, per game, the terminal reward (+1 win / -1 loss). RAE groups by matchup key.

Reuses the cg battle loop (like tools/gen_selfplay) but the pilot side SAMPLES from
softmax(scores / temperature) instead of argmax, so the policy explores. The opponent is
engine_v2 (every agents/*.py runs it), which is also the eval opponent and the baseline the
policy is chasing -- so the reward measures the goal directly.

Sequential by design: the cross-encoder batches a decision's candidates into ONE forward, and
32 games / 2461 decisions measured 54 s, i.e. ~21 min for a 768-game round. The decoder-era
batched inference server (many CPU workers + a GPU server grouping by LoRA adapter) existed
to make a decoder affordable and is quarantined in tools/_legacy_decoder/.

Output: one .jsonl.gz of decision records + a manifest of game rewards, consumed by
tools/rl_advantage.py -> tools/rl_train.py.
"""
import argparse
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

import library                                            # noqa: E402
from lm.serialize import serialize_stateless, multipick_substate, STOP  # noqa: E402
from lm.actions import encode_option                      # noqa: E402
from cg.game import battle_start, battle_select, battle_finish  # noqa: E402

_ACT = "[ACT]\n"


def load_scoring(model_dir, maxlen=640):
    """Scoring model for the CROSS-ENCODER policy (AutoModelForSequenceClassification).

    Replaces the decoder + LoRA loader. The policy definition is unchanged --
    pi(a|s) = softmax over the LEGAL candidates of their scores -- only the way a candidate
    gets its score changes: a decoder needed the mean token logprob of the candidate string,
    the cross-encoder emits one scalar logit per (state, candidate) pair. That is exactly the
    quantity train_rerank.py already optimises with a listwise softmax-CE, so SFT and RL share
    one policy parameterisation and the connection needs no reinterpretation.

    Full fine-tune, so there is no adapter and no set_adapter batching: one model dir in,
    one model on the GPU.
    """
    from eval_rerank import RerankerScoringModel     # the SAME scorer eval_rerank measures
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, trust_remote_code=True, dtype=torch.bfloat16).to("cuda")
    return RerankerScoringModel(model, tok, max_len=maxlen)


def make_serializer(deck_ids, deck_name):
    """serialize_stateless bound to the pilot's deck AND the checkpoint's prompt format.

    rl_config.PROMPT_FMT is the single source of truth; passing the deck ids and the deck
    NAME is not optional (they render DECK[...] and `ID ME d_x a_y`). The pre-existing code
    here called serialize_stateless(obs) bare, which renders the legacy full-rules glossary,
    no DECK[] and no ID -- a ~838-token prompt for a model trained on ~245. Under RL that
    does not just score badly, it puts the gradient on states the policy will never see."""
    import rl_config
    fmt = dict(rl_config.PROMPT_FMT)
    return lambda o: serialize_stateless(o, deck_ids=deck_ids, deck_name=deck_name, **fmt)


def _softmax_sample(scores, temp, rng):
    """Sample an index ~ softmax(scores / temp). Returns (idx, logprob_of_choice)."""
    if temp <= 1e-6:
        i = max(range(len(scores)), key=lambda k: scores[k])
        return i, 0.0
    m = max(scores)
    ex = [math.exp((s - m) / temp) for s in scores]
    Z = sum(ex)
    probs = [e / Z for e in ex]
    r = rng.random()
    acc = 0.0
    for i, pr in enumerate(probs):
        acc += pr
        if r <= acc:
            return i, math.log(max(pr, 1e-12))
    return len(probs) - 1, math.log(max(probs[-1], 1e-12))


def _sampling_pick(model, obs, temp, rng, records, matchup, ser):
    """One decision by the LM policy, SAMPLING. Appends a decision record and returns
    the pick (list[int]) — mirrors lm/agent._score_pick but stochastic + logged.

    ``ser`` is make_serializer(...) for THIS pilot: it carries the deck ids, the deck name
    and rl_config.PROMPT_FMT, so the logged prompt is byte-identical to what the deployed
    agent would build. rl_train re-scores these prompts, so any drift here silently trains
    the policy on a different input distribution than it plays on."""
    sel = obs["select"]
    opts = sel.get("option") or []
    lo = sel.get("minCount", 1) or 0
    hi = sel.get("maxCount", 1) or 1

    def _log(prompt, cands, chosen, scores, logp):
        records.append(dict(matchup=matchup, prompt=prompt, cands=cands,
                            chosen=chosen, scores=scores, old_logp=logp))

    if lo == 1 and hi == 1:
        prompt = _ACT + ser(obs)
        cands = [encode_option(o, obs) for o in opts]
        scores = model.score(prompt, cands, obs)
        if not scores or len(scores) != len(cands):
            return None
        j, logp = _softmax_sample(scores, temp, rng)
        _log(prompt, cands, j, scores, logp)
        return [j]
    # multi-pick / optional: one at a time (same builder as training/inference)
    picked = []
    while len(picked) < hi:
        sub, remaining, allow_stop = multipick_substate(obs, picked)
        if not remaining:
            break
        prompt = _ACT + ser(sub)
        cands = [encode_option(opts[i], obs) for i in remaining]
        if allow_stop:
            cands = cands + [STOP]
        scores = model.score(prompt, cands, obs)
        if not scores or len(scores) != len(cands):
            return None
        j, logp = _softmax_sample(scores, temp, rng)
        _log(prompt, cands, j, scores, logp)
        if allow_stop and j == len(cands) - 1:
            break
        picked.append(remaining[j])
    return picked if len(picked) >= lo else None


def _mk_opp_agent(opp_deck, opp_profile, opp_model, opp_name):
    """Opponent side: engine_v2 (opp_model=None) or a frozen LM checkpoint.

    The format kwargs are NOT optional even though the opponent is frozen. This call used to
    be `make_lm_agent(deck, profile, model)`, whose defaults are glossary="full",
    deck_name=None, deck_mode="static" -- an ~838-token prompt with the legacy card-rules
    header and no ID segment, for a model trained on ~245. It was inert only because the
    opponent was always engine_v2 (model=None ignores the prompt entirely); switching on
    self-play would have silently crippled the opponent and made every self-play win
    meaningless. Same class of bug as main.py and eval_rerank before it."""
    from lm.agent import make_lm_agent
    import rl_config
    if opp_model is None:
        return make_lm_agent(opp_deck, opp_profile, None)      # engine_v2: prompt unused
    return make_lm_agent(opp_deck, opp_profile, opp_model,
                         deck_name=opp_name, **rl_config.PROMPT_FMT)


def _branch_weight(cur, yi, n_opts):
    """How worth branching is this state? Measured over 1,540 branch points (see
    engine-native-search-api): the per-candidate value spread is 0.000 at 1 or 0 prizes left
    (the game is already decided), peaks at 0.375 with 2 prizes, and grows with the number of
    legal options. Spreading branch points uniformly spends most of them where no choice
    changes the outcome."""
    me = (cur.get("players") or [{}, {}])[yi]
    prizes = len(me.get("prize") or [])
    if prizes <= 1:
        return 0.0                       # decided: measured signal was exactly zero
    w = 2.0 if prizes <= 4 else 1.0      # 2-4 prizes carries ~2x the signal of the opening
    if n_opts >= 7:
        w *= 1.3
    elif n_opts <= 3:
        w *= 0.7
    return w


def play_one(pilot, opp, pilot_model, opp_model, temp, rng, records,
             profiles, pilot_first, max_steps=4000, branch=None):
    """Play a single game: pilot (LM, sampling) vs opp (mix). Returns +1/-1/None for the
    PILOT and the count of pilot decisions logged (for MARS turn indexing).

    ``branch`` (optional) enables DECISION-LEVEL groups: at a few states it re-plays the same
    position with each of the top candidates and records where each one ends up, so the update
    can credit THAT decision instead of splitting one game scalar over ~70 of them. Costs no
    GPU -- the counterfactuals run on the engine's native search tree with engine_v2 driving
    the playouts."""
    from lm.agent import make_lm_agent
    d_pilot = [int(x) for x in open(library_deck_path(pilot))]
    d_opp = [int(x) for x in open(library_deck_path(opp))]
    opp_agent = _mk_opp_agent(opp, profiles.get(opp), opp_model, opp)
    ser = make_serializer(d_pilot, pilot)      # deck ids + deck NAME + PROMPT_FMT
    matchup = f"{pilot}__vs__{opp}"
    n0 = len(records)
    d0, d1 = (d_pilot, d_opp) if pilot_first else (d_opp, d_pilot)
    pilot_i = 0 if pilot_first else 1
    br_left = int(branch.get("per_game", 0)) if branch else 0
    br_me = br_op = None
    obs, _ = battle_start(d0, d1)
    if obs is None:
        return None, 0
    try:
        for _ in range(max_steps):
            cur = obs.get("current")
            if cur is None:
                return None, len(records) - n0
            if cur.get("result", -1) != -1:
                r = cur["result"]                          # winner index
                win = 1 if r == pilot_i else -1
                return win, len(records) - n0
            sel = obs.get("select")
            if sel is None:
                return None, len(records) - n0
            yi = cur["yourIndex"]
            if yi == pilot_i:
                opts = sel.get("option") or []
                if sel is None or (len(opts) < 2):
                    choice = _forced(obs, pilot, profiles)  # forced/trivial -> engine
                else:
                    single = (sel.get("minCount", 1) == 1 and sel.get("maxCount", 1) == 1)
                    n_before = len(records)
                    choice = _sampling_pick(pilot_model, obs, temp, rng, records, matchup, ser)
                    if choice is None:
                        choice = _forced(obs, pilot, profiles)
                    elif branch and br_left > 0 and single and len(records) == n_before + 1:
                        w = _branch_weight(cur, yi, len(opts))
                        if w > 0 and rng.random() < min(1.0, branch["rate"] * w):
                            if br_me is None:
                                br_me = make_lm_agent(pilot, profiles.get(pilot), None)
                                br_op = make_lm_agent(opp, profiles.get(opp), None)
                            q = _branch_qvals(obs, d_pilot, d_opp, pilot_i, records[-1],
                                              br_me, br_op, branch, rng)
                            if q is not None:
                                records[-1]["qvals"] = q
                                br_left -= 1
            else:
                choice = opp_agent(obs)
            obs = battle_select(choice)
        return None, len(records) - n0
    finally:
        battle_finish()


def _branch_qvals(obs, d_pilot, d_opp, pilot_i, rec, agent_me, agent_opp, branch, rng):
    """Playout value for the top-K candidates of the decision just logged in ``rec``.

    Returns a list aligned to rec["cands"] with a float for each branched candidate and None
    elsewhere, or None if this state could not be branched. Failures are silent-but-skipped by
    design: rl_branch raises when the visible-card accounting does not reconcile (~3.6% of
    decisions), and a wrong determinization would be worse than no branch at all."""
    try:
        import rl_branch
    except Exception:
        return None
    scores = rec.get("scores") or []
    if len(scores) < 2:
        return None
    k = min(int(branch.get("k", 4)), len(scores))
    # branch what the policy is actually choosing between: its own top-K by score. The
    # sampled action is included so the group always contains the move that was played.
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    if rec.get("chosen") is not None and rec["chosen"] not in order:
        order[-1] = rec["chosen"]
    try:
        qs = rl_branch.branch_values(
            obs, d_pilot, d_opp, pilot_i, [[i] for i in order], agent_me, agent_opp,
            n_playouts=int(branch.get("playouts", 4)), rng=rng)
    except Exception:
        return None
    out = [None] * len(scores)
    got = 0
    for i, q in zip(order, qs):
        if q is not None:
            out[i] = q
            got += 1
    return out if got >= 2 else None


_FORCED_CACHE = {}


def _forced(obs, deck, profiles):
    from lm.agent import make_lm_agent
    if deck not in _FORCED_CACHE:
        _FORCED_CACHE[deck] = make_lm_agent(deck, profiles.get(deck), None)  # engine_v2
    return _FORCED_CACHE[deck](obs)


def library_deck_path(name):
    return os.path.join(ROOT, "decks", name + ".csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, help="A|B|C (tools/rl_config.py)")
    ap.add_argument("--target", default="", help="Stage-C pilot override")
    ap.add_argument("--model", required=True,
                    help="policy checkpoint dir (the SFT reranker, or the last RL round). "
                         "Full weights -- there is no base+adapter split any more.")
    ap.add_argument("--opp-model", default="",
                    help="frozen policy dir to play the lm_selfplay share of games. Required "
                         "whenever the effective heuristic fraction is < 1.0. Held FIXED for "
                         "the stage (not the learning policy) so the target does not move "
                         "under the learner within a stage.")
    ap.add_argument("--heuristic-frac", type=float, default=-1.0,
                    help="override the stage's heuristic opponent fraction (loop anneals this)")
    ap.add_argument("--temperature", type=float, default=-1.0, help="override stage temp")
    ap.add_argument("--matchups", type=int, default=0, help="cap number of (pilot,opp) pairs (0=all)")
    ap.add_argument("--out", required=True, help="output .jsonl.gz of decision records")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1,
                    help="DATA-PARALLEL rollout: split the matchups across N instances. All "
                         "shards MUST share --seed + --matchups so they build the SAME full pair "
                         "list, then each takes a DISJOINT slice; their outputs are merged "
                         "(tools/merge_rollouts.py) into one on-policy rollout for the GRPO step.")
    ap.add_argument("--shard", type=int, default=0, help="this instance's shard index [0, nshards)")
    ap.add_argument("--branch-per-game", type=int, default=0,
                    help="DECISION-LEVEL groups: how many decisions per game to re-play with "
                         "each of the top candidates, so the update can credit that decision "
                         "instead of splitting one game scalar over ~70 of them. 0 = off "
                         "(pure game-level GRPO, the configuration that plateaued). Costs no "
                         "GPU: the counterfactuals run on the engine's search tree with "
                         "engine_v2 playouts, ~0.09 s per branch point per playout round.")
    ap.add_argument("--branch-k", type=int, default=4,
                    help="candidates compared at a branch point (the policy's top-K by score, "
                         "with the sampled action forced in)")
    ap.add_argument("--branch-playouts", type=int, default=4,
                    help="scenarios per branch point; each is a fresh determinization shared "
                         "by all K candidates")
    ap.add_argument("--branch-rate", type=float, default=0.12,
                    help="base acceptance per eligible decision, scaled by _branch_weight "
                         "(prizes left, option count). --branch-per-game is the hard cap.")
    args = ap.parse_args()

    import rl_config
    cfg = rl_config.stage(args.stage, args.target or None)
    temp = args.temperature if args.temperature >= 0 else cfg["temperature"]
    heur = args.heuristic_frac if args.heuristic_frac >= 0 else cfg["opp_agent_mix"]["heuristic"]
    gpm = cfg["games_per_matchup"]
    rng = random.Random(args.seed)

    profiles = json.load(open(os.path.join(ROOT, "agents", "tuning.json")))

    pilots = cfg["pilots"]
    opp_w = {d: w for d, w in cfg["opponents"].items() if w > 0}
    opps = list(opp_w)
    # CURRICULUM sampling (tools/rl_ratings): Stage A "matched" biases pairs to ~50% expected
    # winrate so weak decks get contrastive, learnable games; Stage B/C "weighted" samples by
    # pilot_weights x opponent weights (focus LEARNING on target/live vs the live field).
    mode = cfg.get("sampling", "uniform")
    n = args.matchups or 0
    if mode in ("matched", "weighted"):
        import rl_ratings
        strength, matrix = rl_ratings.load() if mode == "matched" else ({}, {})
        pairs = rl_ratings.sample_pairs(
            pilots, opps, n, rng,
            pilot_w=cfg.get("pilot_weights"), opp_w=opp_w,
            match=(mode == "matched"), strength=strength, matrix=matrix,
            target_wr=cfg.get("match_target_wr", 50.0))
    else:
        pairs = [(p, o) for p in pilots for o in opps if p != o]
        if n:
            pairs = rng.sample(pairs, min(n, len(pairs)))
    full = len(pairs)
    if args.nshards > 1:
        pairs = pairs[args.shard::args.nshards]     # disjoint slice; same seed => same `full` list
    print(f"rl_rollout: stage {args.stage} sampling={mode} -> {len(pairs)}/{full} matchups "
          f"(shard {args.shard}/{args.nshards}, gpm={gpm}) heur={heur:.2f} temp={temp:.2f}", flush=True)

    if heur < 1.0 and not args.opp_model:
        raise SystemExit(f"opponent mix wants {1 - heur:.0%} LM self-play but --opp-model is "
                         "empty; pass a frozen policy dir, or --heuristic-frac 1.0.")
    # Sequential: one game at a time, batched only over a decision's candidates. Two resident
    # cross-encoders at most (learner + frozen opponent, 294 MB each) -- the decoder needed a
    # LoRA set_adapter dance for this; full weights just load twice.
    pilot_model = load_scoring(args.model)
    opp_model = load_scoring(args.opp_model) if (heur < 1.0 and args.opp_model) else None
    branch = None
    if args.branch_per_game > 0:
        branch = dict(per_game=args.branch_per_game, k=args.branch_k,
                      playouts=args.branch_playouts, rate=args.branch_rate)
        print(f"rl_rollout: decision-level groups ON -- up to {args.branch_per_game}/game, "
              f"K={args.branch_k}, {args.branch_playouts} scenarios each", flush=True)
    records, game_rewards = [], []
    for (p, o) in pairs:
        for g in range(gpm):
            first = (g % 2 == 0)
            use_heur = (opp_model is None) or (rng.random() < heur)
            om = None if use_heur else opp_model
            r, n = play_one(p, o, pilot_model, om, temp, rng, records, profiles, first,
                            branch=branch)
            if r is not None:
                game_rewards.append(dict(matchup=f"{p}__vs__{o}", reward=r, n_decisions=n,
                                         opp_kind="heuristic" if use_heur else "lm"))
    with gzip.open(args.out, "wt") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    with open(args.out + ".rewards.json", "w") as f:
        json.dump(game_rewards, f)
    wins = sum(1 for gr in game_rewards if gr["reward"] > 0)
    n_lm = sum(1 for gr in game_rewards if gr["opp_kind"] == "lm")
    print(f"rollout: {len(game_rewards)} games, {len(records)} decisions, "
          f"pilot winrate {wins/max(1,len(game_rewards)):.1%}, temp={temp}, "
          f"heur_frac={heur:.2f} (realised: {len(game_rewards)-n_lm} engine_v2 / {n_lm} LM)",
          flush=True)
    if branch:
        nb = sum(1 for r in records if r.get("qvals"))
        spread = [max(v for v in r["qvals"] if v is not None)
                  - min(v for v in r["qvals"] if v is not None)
                  for r in records if r.get("qvals")]
        print(f"rollout: {nb} branched decisions ({nb/max(1,len(game_rewards)):.1f}/game), "
              f"mean candidate spread {sum(spread)/max(1,len(spread)):.3f}", flush=True)


if __name__ == "__main__":
    main()
