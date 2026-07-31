"""Decision-level branch evaluation on the engine's native search tree.

WHY THIS EXISTS
    GRPO here gives every decision in a game the SAME advantage (terminal reward minus a
    matchup baseline), so one scalar has to explain ~70 decisions. That is the measured
    cause of the RL plateau: 8,064 training games moved the gate 0.0pt
    (rl-stage-a-plateau-diagnosis). The fix is a group of actions branched from ONE state,
    which is what `cg.api`'s search tree provides and nothing in the repo used.

WHY THE SEARCH TREE AND NOT A REPLAY
    The native engine's RNG is not reachable from Python (no seed symbol in libcg.so; a
    Python-randomness-free agent produces a different game every run, with or without
    random.seed). So a state cannot be re-reached by replaying. The search tree branches
    from the live state instead, and because the K children share one root they also share
    one determinization of the hidden cards -- the variance reduction that seeding would
    have given, for free.

THE DETERMINIZATION INVARIANT
    search_begin only validates COUNTS, so "it returned a node" is not evidence of
    correctness. What we can check exactly: everything not visible is either deck or prize
    (ours) / deck, hand or prize (theirs). If our reconstruction is right those totals match
    to the card. `unseen_multisets` raises when they do not, so a missed zone is a loud
    failure instead of a silently wrong branch.
"""
import collections
import ctypes
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cg.api as api            # noqa: E402
import cg.sim as sim            # noqa: E402


class DeterminizationError(RuntimeError):
    """The visible-card accounting did not reconcile; the branch would be wrong."""


def _cid(x):
    if isinstance(x, dict):
        return x.get("id")
    return x if isinstance(x, int) else None


def _board_ids(pl):
    """Every card of this player's that is FACE UP: board, attached tools and energy,
    the evolution stack underneath, and discard. Deliberately excludes hand (caller
    decides -- ours is visible, theirs is not) and the hidden zones.

    Two traps, both measured (the naive version reconciled on only 17% of decisions):
      * `preEvolution` holds the cards buried under an evolved Pokemon. They are real
        cards off the decklist and nothing else in the observation mentions them, so
        missing them makes the unseen pool grow by one per evolution as the game runs.
      * prefer `energyCards` (card objects) over `energies` (one entry per energy UNIT),
        or a special energy providing two units is counted as two cards.
    """
    out = []
    for z in ("active", "bench"):
        for x in (pl.get(z) or []):
            if not x:
                continue
            i = _cid(x)
            if i is not None:
                out.append(i)
            for group in (x.get("tools"),
                          x.get("energyCards") if x.get("energyCards") is not None
                          else x.get("energies"),
                          x.get("preEvolution")):
                for c in (group or []):
                    i = _cid(c)
                    if i is not None:
                        out.append(i)
    for d in (pl.get("discard") or []):
        i = _cid(d)
        if i is not None:
            out.append(i)
    return out


def _subtract(pool, used, who):
    c = collections.Counter(pool)
    for i in used:
        if c[i] <= 0:
            raise DeterminizationError(
                "%s: card %s seen more often than the decklist contains" % (who, i))
        c[i] -= 1
    return list(c.elements())


def unseen_multisets(obs, my_deck, opp_deck):
    """(my_unseen, opp_unseen) -- the cards each player could still be holding.

    my_unseen  == my deck + my prizes           (my hand is visible to me)
    opp_unseen == their deck + hand + prizes

    Raises DeterminizationError if the counts do not reconcile exactly.
    """
    cur = obs.get("current") or {}
    players = cur.get("players") or []
    yi = cur.get("yourIndex", 0)
    if len(players) != 2:
        raise DeterminizationError("expected 2 players, got %d" % len(players))
    me, opp = players[yi], players[1 - yi]

    my_used = _board_ids(me) + [_cid(h) for h in (me.get("hand") or [])]
    opp_used = _board_ids(opp)

    # The stadium sits outside both players' zones but came off one of the decks. It
    # carries `playerIndex`, so charge it to its real owner -- guessing by "who can still
    # afford it" mis-attributes every stadium the opponent played and shrinks our pool by
    # exactly one.
    for s in (cur.get("stadium") or []):
        i = _cid(s)
        if i is None:
            continue
        owner = s.get("playerIndex") if isinstance(s, dict) else None
        (my_used if owner == yi else opp_used).append(i)

    my_unseen = _subtract(my_deck, [i for i in my_used if i is not None], "me")
    opp_unseen = _subtract(opp_deck, [i for i in opp_used if i is not None], "opp")

    want_me = int(me.get("deckCount", 0)) + len(me.get("prize") or [])
    want_opp = (int(opp.get("deckCount", 0)) + int(opp.get("handCount", 0))
                + len(opp.get("prize") or []))
    # check both sides before raising: reporting only the first hides the other's error
    bad = []
    if len(my_unseen) != want_me:
        bad.append("my unseen %d != deck %d + prize %d"
                   % (len(my_unseen), me.get("deckCount", 0), len(me.get("prize") or [])))
    if len(opp_unseen) != want_opp:
        bad.append("opp unseen %d != deck %d + hand %d + prize %d"
                   % (len(opp_unseen), opp.get("deckCount", 0), opp.get("handCount", 0),
                      len(opp.get("prize") or [])))
    if bad:
        raise DeterminizationError("; ".join(bad))
    return my_unseen, opp_unseen


def _raw_step(search_id, select):
    """lib.SearchStep without api.py's dataclass layer: the native JSON already carries a
    dict observation shaped exactly like the one agents consume, so engine_v2 needs no
    adapter."""
    arr = (ctypes.c_int * len(select))(*select)
    bs = sim.lib.SearchStep(api.agent_ptr, search_id, arr, len(select))
    return json.loads(bs.decode())


def _playout(state, pilot_i, agent_me, agent_opp, max_steps=4000):
    """Drive a branch to a terminal result with engine_v2 on both sides.
    Returns +1 / -1 for the PILOT, or None if the branch did not resolve."""
    steps = 0
    while steps < max_steps:
        ob = state.get("observation") or {}
        cur = ob.get("current")
        if cur is None:
            return None
        r = cur.get("result", -1)
        if r != -1:
            return 1 if r == pilot_i else -1
        if not ob.get("select"):
            return None
        agent = agent_me if cur.get("yourIndex") == pilot_i else agent_opp
        try:
            choice = agent(ob)
        except Exception:
            return None
        nxt = _raw_step(state["searchId"], choice)
        if nxt.get("error", 0) != 0 or not nxt.get("state"):
            return None
        state = nxt["state"]
        steps += 1
    return None


def branch_values(obs, my_deck, opp_deck, pilot_i, selections,
                  agent_me, agent_opp, n_playouts=1, rng=None):
    """Q for each candidate selection, averaged over n_playouts SCENARIOS.

    One scenario = one determinization (a shuffle of the unseen pool) + one playout of every
    candidate from that scenario's root. So the hidden cards are COMMON to all candidates
    within a scenario -- the comparison is not polluted by one branch drawing a better deck
    than another -- while averaging across scenarios integrates out the hidden state we do not
    know. Re-stepping a single root instead would be the wrong axis: the outcome still varies
    (measured: 5 of 6 branch points gave 2 distinct results over 8 repeats), but every repeat
    would keep the same hidden cards, so the average stays conditional on one guess.

    `selections` is a list of raw selections (list[int]) -- the same objects an agent returns.
    Returns a list aligned to `selections`, with None where a candidate never resolved, so the
    caller can drop it rather than score it as a loss.
    """
    my_unseen, opp_unseen = unseen_multisets(obs, my_deck, opp_deck)
    cur = obs["current"]
    yi = cur["yourIndex"]
    opp_active = cur["players"][1 - yi].get("active") or []
    need_active = len(opp_active) > 0 and opp_active[0] is None

    vals = [[] for _ in selections]
    for _ in range(max(1, n_playouts)):
        mu, ou = list(my_unseen), list(opp_unseen)
        if rng is not None:
            rng.shuffle(mu)
            rng.shuffle(ou)
        # only needed when their Active is face down; search_begin ignores it otherwise
        active_guess = [ou[0]] if need_active and ou else []
        try:
            root = api.search_begin(api.to_observation_class(obs),
                                    mu, mu, ou, ou, ou, active_guess)
        except Exception:
            continue
        try:
            for i, sel in enumerate(selections):
                step = _raw_step(root.searchId, sel)
                if step.get("error", 0) != 0 or not step.get("state"):
                    continue
                v = _playout(step["state"], pilot_i, agent_me, agent_opp)
                if v is not None:
                    vals[i].append(v)
        finally:
            api.search_end()
    return [sum(v) / len(v) if v else None for v in vals]
