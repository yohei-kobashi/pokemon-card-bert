"""Generate large-scale heuristic self-play data for LM-agent training.

Drives every deck's heuristic agent against every other through the cg C-library
and captures, for each in-game selection, a lossless structured record. The text
serialization is intentionally NOT done here -- a later ``build_sft.py`` renders
these records into training text. Because the cg RNG is not seedable (games are
not reproducible), capture must be faithful and irreversible-safe.

Output layout (see docs/ml_agent_plan.md sec.5):

    data/selfplay/<tag>/<deckA>__vs__<deckB>.jsonl.gz   # game header + step lines
    data/selfplay/<tag>/manifest.jsonl                  # one light line per game

Each matchup file is written by exactly one worker process (no cross-process
contention); workers return only the small manifest rows over IPC.

Usage:
    python tools/gen_selfplay.py                        # all decks, 20 games/pair
    python tools/gen_selfplay.py --games 40 --tag v1
    python tools/gen_selfplay.py --decks mega_lucario,crustle --mirror
    python tools/gen_selfplay.py --lean                 # prune None option fields
"""
import argparse
import collections
import gzip
import itertools
import json
import os
import random
import sys
import time
from datetime import datetime
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.dirname(__file__)):
    if p not in sys.path:
        sys.path.insert(0, p)

import library  # noqa: E402
from battle_log import load_agent  # noqa: E402
from cg.game import battle_start, battle_select, battle_finish  # noqa: E402

# Engine metadata used only by the exploration blunder-floor (案B). Optional: if
# engine_v2 can't be imported, the floor degrades to option-type checks only.
try:
    from agents.engine_v2 import _ATTACKS as _EV2_ATTACKS, _atk_value as _ev2_atk_value  # noqa: E402
except Exception:  # pragma: no cover
    _EV2_ATTACKS, _ev2_atk_value = {}, None

# Exploration draws only from the engine's top-K options (案A). Set from
# --explore-topk in main() BEFORE the worker Pool is forked (children inherit it).
_EXPLORE_TOPK = 4

PRIZE_START = 6  # standard prize count; taken = PRIZE_START - remaining

# Per-process cache of (agent_fn, deck_ids), keyed by deck name.
_CACHE = {}


ENGINE = os.environ.get("SELFPLAY_ENGINE", "v2")


def _load(name):
    if name not in _CACHE:
        deck = library.read_deck(name)
        if ENGINE == "v2":
            # v2.4 engine pilots (pipeline-validated: 52/52 parity-or-better
            # vs legacy) — the data source for LM training.
            import json as _json
            from agents import engine_v2
            tun = _json.load(open(os.path.join(ROOT, "agents", "tuning.json")))
            pol = engine_v2.make_policy(deck, tun.get(name, {}))
            _CACHE[name] = (pol.act, deck)
        else:
            _CACHE[name] = (load_agent(name), deck)
    return _CACHE[name]


_TRAINER_POOL = None


def _trainer_pool():
    """Every ordinary Trainer card any deck runs -- the substitution alphabet.

    Trainers only: swapping a Pokemon breaks evolution lines and swapping energy breaks the
    type the attackers need, either of which stops the deck functioning and makes the
    engine's play unrepresentative. ACE SPECs are excluded because a deck may hold exactly
    one, so swapping one in can produce an ILLEGAL list.

    Replacements are drawn PROPORTIONAL TO HOW MANY DECKS RUN THE CARD, so a perturbed list
    looks like a plausible variant rather than a pile of niche cards: staples keep appearing
    often enough for the model to learn their roles, which is the point of the exercise, and
    the result is closer to the real thing we want robustness to -- a competitor running a
    slightly different build."""
    global _TRAINER_POOL
    if _TRAINER_POOL is None:
        from lm import vocab
        n_decks = collections.Counter()
        for nm in library.list_decks():
            for cid in set(library.read_deck(nm)):
                c = vocab.card(cid)
                if c and c.cardType in (1, 2, 3, 4) and not getattr(c, "aceSpec", False):
                    n_decks[cid] += 1
        ids = sorted(n_decks)
        _TRAINER_POOL = (ids, [n_decks[c] for c in ids])
    return _TRAINER_POOL


def _perturb(deck, rng, n_swap, roles=None):
    """Swap ``n_swap`` single Trainer copies for other Trainers. Returns a new 60-card list.

    WHY THIS EXISTS: the LM is supposed to read ``DECK[...]``, and with 62 FIXED decklists it
    never will -- ``ID ME d_alakazam`` determines the 60 cards exactly, so a 3-token lookup is
    a sufficient statistic for everything the teacher derives from the deck (engine_v2 uses
    the list only through ``infer_roles`` and its tier set, both constants per deck). Making
    the same deck NAME correspond to many lists is what turns the card tokens into
    information. engine_v2 recomputes roles from the card database, so it pilots a perturbed
    list correctly and its labels genuinely change with the swap.

    Only FLEX slots are removed -- cards whose tuning.json card_roles entry is ``tech`` or
    ``filler``, plus unlisted ones. Real lists vary in exactly those slots; a competitor does
    not cut their draw engine. Removing ``win``/``engine``/``line`` cards instead produces a
    deck that cannot execute, and then the engine's labels describe a broken deck rather than
    the archetype (measured: an unrestricted 3-swap took mega_lucario from 58.3% to 31.7%).

    Keeps 60 cards and the 4-copy limit; touches nothing but Trainers."""
    pool, weight = _trainer_pool()
    out = list(deck)
    from lm import vocab
    flex = {"tech", "filler"}
    for _ in range(n_swap):
        removable = [i for i, cid in enumerate(out)
                     if (c := vocab.card(cid)) and c.cardType in (1, 2, 3, 4)
                     and not getattr(c, "aceSpec", False)
                     and (roles or {}).get(str(cid), "tech") in flex]
        if not removable:
            break
        counts = {}
        for cid in out:
            counts[cid] = counts.get(cid, 0) + 1
        add = [(cid, w) for cid, w in zip(pool, weight) if counts.get(cid, 0) < 4]
        if not add:
            break
        out.pop(rng.choice(removable))
        out.append(rng.choices([c for c, _ in add], weights=[w for _, w in add])[0])
    return out


def _pilot(name, rng=None, n_swap=0):
    """(act, deck) for ``name``; a perturbed variant gets its OWN policy because
    engine_v2 computes roles/tiers from the deck at construction."""
    if not n_swap or rng is None:
        return _load(name)
    _act, base = _load(name)
    import json as _json
    tun = _json.load(open(os.path.join(ROOT, "agents", "tuning.json")))
    prof = tun.get(name, {})
    deck = _perturb(base, rng, n_swap, prof.get("card_roles"))
    if ENGINE == "v2":
        from agents import engine_v2
        return engine_v2.make_policy(deck, prof).act, deck
    return load_agent(name), deck


def _random_legal(sel, rng):
    """A random LEGAL selection for this select (respects min/maxCount)."""
    opts = sel.get("option") or []
    n = len(opts)
    if n == 0:
        return None
    hi = min(sel.get("maxCount", 1) or 1, n)
    lo = min(sel.get("minCount", 1) or 0, hi)
    k = rng.randint(lo, hi)
    return sorted(rng.sample(range(n), k)) if k > 0 else []


def _rank_by_removal(agent, obs, k):
    """Rank MAIN options best-first: ask the agent, drop its pick, renumber, ask
    again (up to k times). Returns a best-first list of ORIGINAL option indices.

    Relies on the policy being a pure function of obs (verified: deterministic and
    stateless across interleaved calls), so these extra calls never perturb the
    live game's later decisions. Uses a shallow copy with a fresh option list, so
    the real obs is untouched.
    """
    sel = obs.get("select") or {}
    opts = sel.get("option") or []
    remaining = list(range(len(opts)))
    ranking = []
    for _ in range(min(k, len(opts))):
        if not remaining:
            break
        sub = dict(obs)
        sub_sel = dict(sel)
        sub_sel["option"] = [opts[j] for j in remaining]
        sub["select"] = sub_sel
        try:
            pick = agent(sub)
        except Exception:
            break
        if not pick or not (0 <= pick[0] < len(remaining)):
            break
        ranking.append(remaining.pop(pick[0]))
    return ranking


def _is_damaging_attack(opt):
    """True if this option is an attack that deals damage (scaling attacks
    included via _atk_value). Conservative when the engine is unavailable."""
    if opt.get("type") != 13:
        return False
    if _ev2_atk_value is None:
        return True
    a = _EV2_ATTACKS.get(opt.get("attackId"))
    return a is not None and _ev2_atk_value(a) > 0


def _plausible_legal(sel, choice, obs, rng, agent):
    """A random EXPLORATION move drawn only from PLAUSIBLE options, so exploration
    visits diverse-but-realistic states instead of executing obvious blunders.

      案A  restrict to the engine's top-K options (_rank_by_removal); the tail
           (clearly-wrong moves the engine rates last) is dropped by construction.
      案B  hard floor: never 'end turn' (type 14) while a damaging attack or an
           energy attach is still on offer -- the catastrophic pass.

    Then remove the heuristic's own choice (we want to explore, not repeat the
    exploit). Returns a legal single-index selection, or None to SKIP exploration
    (execute the heuristic move) when nothing plausible-and-different remains.
    Only single-pick MAIN selects explore; multi-pick selects skip (never risk an
    illegal multi-index set that would forfeit the game).
    """
    opts = sel.get("option") or []
    n = len(opts)
    if n <= 1:
        return None
    if (sel.get("maxCount") or 1) != 1 or not choice or len(choice) != 1:
        return None
    ranking = _rank_by_removal(agent, obs, _EXPLORE_TOPK)
    pool = set(ranking) if ranking else set(range(n))       # 案A (graceful fallback)
    if any(_is_damaging_attack(o) for o in opts) or any(o.get("type") == 8 for o in opts):
        pool = {j for j in pool if opts[j].get("type") != 14}   # 案B floor
    pool.discard(choice[0])
    if not pool:
        return None
    return [rng.choice(sorted(pool))]


def _clean_obs(obs, lean, keep_blob=False):
    """Copy of obs stripped of fields not useful for training.

    Drops ``search_begin_input`` (the search-API blob) UNLESS ``keep_blob``. The v41 prompt
    decodes the engine's hidden effect state out of that blob (lm/hidden.py), so a tag generated
    without it renders v41 as if nothing were modified -- silently, since every fact is optional.
    Measured cost of keeping it: 1,281 bytes/step against a 4,328-byte step record, i.e. +30% on
    a tag that pool_daemon deletes at the end of the round anyway.

    With ``lean``, also drops None-valued keys inside each select option (they dominate the
    Option dict) to shrink the footprint. Never touches state/logs semantics.
    """
    o = dict(obs)
    if not keep_blob:
        o.pop("search_begin_input", None)
    if lean:
        sel = o.get("select")
        if isinstance(sel, dict) and isinstance(sel.get("option"), list):
            sel = dict(sel)
            sel["option"] = [
                {k: v for k, v in opt.items() if v is not None}
                if isinstance(opt, dict) else opt
                for opt in sel["option"]
            ]
            o["select"] = sel
    return o


def _mk_step(o, cur, i):
    """Build a step record from a cleaned obs ``o`` (choice/is_winner added later)."""
    sel = o.get("select") or {}
    opts = sel.get("option") or []
    return {
        "kind": "step",
        "i": i,
        "player": cur.get("yourIndex"),
        "turn": cur.get("turn"),
        "turn_action": cur.get("turnActionCount"),
        "context": sel.get("context"),
        "is_main": sel.get("context") == 0,
        "min": sel.get("minCount"),
        "max": sel.get("maxCount"),
        "n_options": len(opts),
        "obs": o,
    }


def _play_game(order, game_id, max_steps, lean, eps=0.0, perturb=0, perturb_frac=0.0,
               keep_blob=False):
    """Play one battle. ``order`` = (name0, name1) mapping player index -> deck.

    Returns (header_dict, [step_dicts]) or None if the battle failed to start.
    Records a step only if its selection was legal (a forfeiting illegal move is
    not stored; the game closes with end_reason='forfeit').

    ``eps`` > 0 enables exploration: on MAIN selections, with probability eps the
    game EXECUTES a PLAUSIBLE alternative move (``_plausible_legal``: the engine's
    top-K minus obvious blunders) to visit new states, but the recorded label
    (``action``/``chosen``) stays the heuristic's choice (mini-DAgger). So training
    sees diverse-but-realistic boards with good labels, never a blunder-reached one.
    """
    _prng = random.Random()
    n0 = perturb if (perturb and _prng.random() < perturb_frac) else 0
    n1 = perturb if (perturb and _prng.random() < perturb_frac) else 0
    a0, d0 = _pilot(order[0], _prng, n0)
    a1, d1 = _pilot(order[1], _prng, n1)
    if len(d0) != 60 or len(d1) != 60:
        return None
    agents = (a0, a1)
    obs, _sd = battle_start(d0, d1)
    if obs is None:
        return None

    rng = random.Random()
    steps = []
    winner = None
    end_reason = "timeout"
    first_player = -1
    final = obs
    try:
        for _ in range(max_steps):
            cur = obs.get("current")
            if cur is None:
                end_reason = "draw"
                break
            if first_player == -1:
                first_player = cur.get("firstPlayer", -1)
            if cur.get("result", -1) != -1:
                winner = cur["result"]
                end_reason = "result"
                final = obs
                break
            sel = obs.get("select")
            if sel is None:
                end_reason = "draw"
                break
            yi = cur["yourIndex"]
            o = _clean_obs(obs, lean, keep_blob)
            try:
                choice = agents[yi](obs)          # heuristic action = training LABEL
            except Exception:
                winner = 1 - yi
                end_reason = "forfeit"
                break
            # eps-exploration: on MAIN, sometimes EXECUTE a PLAUSIBLE alternative
            # move (engine top-K minus obvious blunders) to visit diverse-but-
            # realistic states; the recorded label stays the heuristic choice.
            executed, explored = choice, False
            if eps > 0 and (obs.get("select") or {}).get("context") == 0 \
                    and rng.random() < eps:
                alt = _plausible_legal(obs["select"], choice, obs, rng, agents[yi])
                if alt is not None and alt != choice:
                    executed, explored = alt, True
            try:
                nxt = battle_select(executed)
            except Exception:
                winner = 1 - yi
                end_reason = "forfeit"
                break
            step = _mk_step(o, cur, len(steps))
            step["action"] = list(choice)
            osel = o.get("select") or {}
            oopts = osel.get("option") or []
            step["chosen"] = [oopts[j] for j in choice if 0 <= j < len(oopts)]
            if explored:
                step["explored"] = True
                step["executed"] = list(executed)
            steps.append(step)
            obs = nxt
            final = obs
    finally:
        battle_finish()

    fcur = (final or {}).get("current") or {}
    players = fcur.get("players") or []
    prize_remaining = {str(i): len(p.get("prize") or []) for i, p in enumerate(players)}
    deck_remaining = {str(i): p.get("deckCount") for i, p in enumerate(players)}

    for s in steps:
        s["game_id"] = game_id
        s["is_winner"] = None if winner is None else (s["player"] == winner)

    header = {
        "kind": "game",
        "schema": 1,
        "game_id": game_id,
        "decks": {"0": d0, "1": d1},
        "agents": {"0": order[0], "1": order[1]},
        "first_player": first_player,
        "winner": winner,
        "end_reason": end_reason,
        "n_steps": len(steps),
        "prize_remaining": prize_remaining,
        "deck_remaining": deck_remaining,
        "explore_eps": eps,
        # per seat: how many Trainer copies were swapped out of that deck. Non-zero means the
        # logged 60 deliberately differ from decks/<name>, so build_rerank's staleness check
        # must not read it as an outdated decklist.
        "perturb": {"0": n0, "1": n1},
    }
    return header, steps


def _play_pair(task):
    """Worker: play all games of one matchup, write its .jsonl.gz, return manifest rows.

    Alternates which deck is player-index 0 across games (as arena.match does).
    The actual first player is decided by the engine and recorded per game.
    """
    (nameA, nameB, games, outdir, lean, max_steps, rel, eps, perturb, perturb_frac,
     keep_blob) = task
    fname = f"{nameA}__vs__{nameB}.jsonl.gz"
    path = os.path.join(outdir, fname)
    rows = []
    n_games = n_steps = 0
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for g in range(games):
            order = (nameA, nameB) if g % 2 == 0 else (nameB, nameA)
            game_id = f"{nameA}__vs__{nameB}#{g:05d}"
            res = _play_game(order, game_id, max_steps, lean, eps, perturb, perturb_frac,
                             keep_blob)
            if res is None:
                continue
            header, steps = res
            f.write(json.dumps(header, separators=(",", ":")) + "\n")
            for s in steps:
                f.write(json.dumps(s, separators=(",", ":")) + "\n")
            n_games += 1
            n_steps += len(steps)
            rows.append({
                "game_id": header["game_id"],
                "file": os.path.join(rel, fname),
                "agents": header["agents"],
                "first_player": header["first_player"],
                "winner": header["winner"],
                "end_reason": header["end_reason"],
                "n_steps": header["n_steps"],
                "prize_remaining": header["prize_remaining"],
                "deck_remaining": header["deck_remaining"],
            })
    return nameA, nameB, n_games, n_steps, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=20, help="games per matchup")
    ap.add_argument("--decks", type=str, default="", help="comma-separated subset")
    ap.add_argument("--mirror", action="store_true",
                    help="also generate self-mirror matchups (deck vs itself)")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--keep-blob", action="store_true",
                    help="keep obs['search_begin_input'] -- REQUIRED for build_rerank --pfmt "
                         "v41, which decodes the engine's hidden effect state out of it. Costs "
                         "+30% tag size on a tag that is deleted at the end of the round.")
    ap.add_argument("--lean", action="store_true",
                    help="prune None-valued option fields to shrink files")
    ap.add_argument("--explore", type=float, default=0.1,
                    help="eps: on MAIN, execute a PLAUSIBLE alternative move with this prob "
                         "to diversify states. The step records BOTH the heuristic choice and "
                         "the executed move; build_sft adopts by the evaluator. 0.10-0.25 typical")
    ap.add_argument("--perturb", type=int, default=0,
                    help="swap this many Trainer copies out of each perturbed deck. With 62 "
                         "FIXED decklists the deck token determines the list, so no model "
                         "will ever read DECK[]'s cards; varying the list under the same NAME "
                         "is what makes them informative. See _perturb.")
    ap.add_argument("--perturb-frac", type=float, default=0.5,
                    help="probability each SEAT is perturbed. Keep well below 1.0: the decks "
                         "we actually ship are the canonical ones and must stay in the data.")
    ap.add_argument("--explore-topk", type=int, default=4,
                    help="exploration draws only from the engine's top-K options (案A). "
                         "Smaller = safer/less diverse; excludes the argmax so K>=2 to explore")
    ap.add_argument("--tag", type=str, default="",
                    help="run subdir under data/selfplay (default: timestamp)")
    ap.add_argument("--out", type=str, default=os.path.join(ROOT, "data", "selfplay"))
    args = ap.parse_args()

    global _EXPLORE_TOPK
    _EXPLORE_TOPK = args.explore_topk       # set before the Pool is forked

    decks = library.list_decks()
    if args.decks:
        want = set(args.decks.split(","))
        decks = [d for d in decks if d in want]
    decks = sorted(decks)

    tag = args.tag or datetime.now().strftime("%Y%m%d-%H%M%S")
    rel = os.path.join("data", "selfplay", tag)
    outdir = os.path.join(args.out, tag)
    os.makedirs(outdir, exist_ok=True)

    pairs = list(itertools.combinations(decks, 2))
    if args.mirror:
        pairs += [(d, d) for d in decks]
    tasks = [(a, b, args.games, outdir, args.lean, args.max_steps, rel, args.explore,
              args.perturb, args.perturb_frac, args.keep_blob) for a, b in pairs]

    print(f"{len(decks)} decks, {len(pairs)} matchups x {args.games} games "
          f"= up to {len(pairs) * args.games} battles on {args.workers} workers "
          f"(explore eps={args.explore}, top-{args.explore_topk})")
    print(f"-> {outdir}")

    manifest_path = os.path.join(outdir, "manifest.jsonl")
    t = time.time()
    done = tot_games = tot_steps = 0
    with open(manifest_path, "w", encoding="utf-8") as mf, \
            Pool(args.workers) as pool:
        for a, b, ng, ns, rows in pool.imap_unordered(_play_pair, tasks):
            for r in rows:
                mf.write(json.dumps(r, separators=(",", ":")) + "\n")
            mf.flush()
            done += 1
            tot_games += ng
            tot_steps += ns
            if done % 20 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} matchups | {tot_games} games "
                      f"| {tot_steps} steps | {time.time() - t:.0f}s", flush=True)

    summary = {
        "tag": tag, "decks": decks, "games_per_pair": args.games,
        "mirror": args.mirror, "matchups": len(pairs), "explore_eps": args.explore,
        "perturb": args.perturb, "perturb_frac": args.perturb_frac,
        "explore_topk": args.explore_topk,
        "games": tot_games, "steps": tot_steps,
        "generated": datetime.now().strftime("%Y%m%d-%H%M%S"),
    }
    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\ndone: {tot_games} games, {tot_steps} steps -> {outdir}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
