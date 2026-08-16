"""The plan-rule wrapper, on the DEPLOY side of the tree.

This code used to live in tools/mirror_match.py, which is a research harness and is not
shippable. That placement was fine while the wrapper was an experiment and fatal the moment it
was adopted: the `def` arm won its gate on 2026-08-11 and there was no way to put it in a
tarball. tools/mirror_match.py now imports make_plan_rule from here, so the arm that was
measured and the arm that ships are the same function rather than two copies that agree today.

`dusk_plan` is imported LAZILY and by three names. In the repo it is tools/dusk_plan.py, reached
either as a top-level module (tools/ on sys.path, how mirror_match runs) or as tools.dusk_plan.
In a submission bundle it is exec'd into sys.modules under its bare name by main.py, next to the
embedded lm/ modules. Resolving it at call time rather than at import time keeps this module
importable in all three, which matters because lm/agent.py must not acquire a hard dependency on
the research tree.
"""
import os


# Rules phrased as "do NOT ..." -- their conformant set is (every option MINUS the forbidden
# ones), so they must SUBTRACT from whatever survives, never be OR'd into it.
#
# The original filter unioned every rule's conformant set. That is right for rules that name the
# move to make and wrong for rules that name the move to avoid: a prohibition's set is ~85% of
# the menu, so OR-ing it back in re-admits exactly the options a positive rule had just excluded.
# Measured on the 19 live games of submission 55445834: a positive rule fired on 74 of our menus,
# a prohibition co-fired on 18 of them (24%), and on all 18 the union was wider than the correct
# answer. Prohibitions intersect; positives union; the two compose.
PROHIBITIONS = frozenset({
    "judge_timing",      # do not Judge while the opponent holds no more cards than you
    "spare_ex_bench",    # do not bench a body worth the prizes they need to win
    "retreat_energy",    # do not retreat a body carrying {R}/{P}
    "stadium_replace",   # do not play a Stadium onto an identical Stadium
    "clops_hold",        # do not fire Dusclops' Cursed Blast while Dusknoir is in hand
    "search_bottom",     # do not search out a card whose pre-evolution is not in play
})


def _plan():
    try:
        import dusk_plan
        return dusk_plan
    except ImportError:
        pass
    from tools import dusk_plan
    return dusk_plan


def make_plan_rule(lm_agent, rules, strict=True, decider=None):
    """Hand the decisions `rules` fire on to the PLAN, not to engine_v2.

    Two decision families are worth taking off the model entirely: damage-counter placement
    (spread_aim) and energy allocation (energy_line, energy_focus). Measured against the chance
    of picking a conformant option at random, s1 barely chooses on the second --

        spread_aim    chance 31.1%   s1 47.5%   (+16.4)
        energy_focus  chance 37.3%   s1 48.5%   (+11.2)
        energy_line   chance 41.2%   s1 46.0%    (+4.8)   <- 16,533 rows, the most frequent

    -- and energy_line at +4.8 over chance on the single most common decision is the same
    finding as [[attach-decisions-at-chance]] reproduced on a different model.

    strict=True   the rule decides: take its highest-weighted conformant option.
    strict=False  the rule FILTERS and the model ranks inside the survivors, by re-asking with
                  the menu restricted. Only for exactly-one picks; a multi-pick menu is left to
                  the model, because rewriting min/maxCount would change what is being asked.

    Two deliberate scope limits, so a reader does not over-credit the arm:
      * multi-pick menus (minCount/maxCount beyond exactly-one) fall back to the model in BOTH
        modes -- rewriting min/maxCount would change what is being asked. The rule pilots the
        exactly-one placements, which is what DAMAGE_COUNTER selects are.
      * strict picks the rule's action IMMEDIATELY on a menu that may also offer benching or
        trainers. None of the deferred actions end the turn (attach / ability / counter
        placement), so the cost is ordering within the turn, and the other plays return on the
        next menu. Attacks are not deferred here, and the plan itself orders those last.

    This is a TEST as much as a policy: the plan says these rules are right, and nothing has
    ever checked that against games. If the rules are wrong, this arm loses.
    """
    dusk_plan = _plan()

    want = [r for r in rules if r in dusk_plan.RULES]
    if len(want) != len(rules):
        raise SystemExit("unknown plan rule in %r (have %s)" % (rules, sorted(dusk_plan.RULES)))

    def f(obs):
        sel = obs.get("select") or {}
        opts = sel.get("option") or []
        if len(opts) < 2:
            return lm_agent(obs)
        try:
            live = dusk_plan.opportunities(obs)
        except Exception:                                      # noqa: BLE001
            return lm_agent(obs)                               # a rule that errors must not pilot
        w = {}
        allowed = None                 # intersection of every prohibition that fired
        for r in want:
            hit = live.get(r)
            if not hit:
                continue
            good, _scope = hit
            good = set(good)
            if not good:
                continue
            if r in PROHIBITIONS:
                allowed = good if allowed is None else (allowed & good)
                continue               # a prohibition subtracts; it never nominates
            for i in good:
                w[i] = w.get(i, 0.0) + dusk_plan.RULES[r][1]
        if allowed is not None:
            if w:
                # A positive rule nominated; keep only its nominees that are not forbidden.
                w = {i: v for i, v in w.items() if i in allowed}
            if not w:
                # Nothing nominated (or everything nominated was forbidden): the prohibition
                # alone decides the menu, and the model ranks freely inside what is left.
                w = {i: 1.0 for i in allowed}
        if not w or len(w) >= len(opts):
            # len(w) >= len(opts) means nothing was actually removed -- re-asking the model with
            # an identical menu would just pay for a second forward pass.
            return lm_agent(obs)
        if decider is not None:
            # The MATCHED control: same triggers, a different decider. Without it a win for the
            # rule cannot be told apart from "anything other than the model is better on these
            # menus", which is a different and much weaker claim.
            return decider(obs)
        lo = sel.get("minCount", 1) or 0
        hi = sel.get("maxCount", 1) or 1
        # "up to 1" (lo 0, hi 1) is the shape of nearly every deck search, and the exactly-one
        # test below rejected all of them -- so a prohibition could fire on a search and change
        # nothing. Restricting the OPTIONS of an up-to-1 menu does not touch min/max and does
        # not remove the right to decline, so the objection does not apply to it.
        _upto1 = os.environ.get("SB_UPTO1", "") not in ("", "0")
        if strict:
            best = max(sorted(w), key=lambda i: w[i])           # sorted() -> ties break low
            if lo <= 1 <= hi:
                return [best]
            return lm_agent(obs)                                # multi-pick: not ours to force
        if not (lo == 1 and hi == 1) and not (_upto1 and lo <= 1 and hi == 1):
            return lm_agent(obs)
        keep = sorted(w)
        if len(keep) == 1:
            return keep
        # Two-level copy, not deepcopy: obs drags the search_begin_input blob along, and the
        # sub-menu path only ever READS the shared parts. Only the containers being replaced
        # are duplicated.
        sub = dict(obs)
        sub["select"] = dict(sel)
        sub["select"]["option"] = [opts[i] for i in keep]
        pick = lm_agent(sub)
        if not pick:
            # On an up-to-1 menu an empty pick is a legal DECLINE, not a failure to answer;
            # forcing keep[0] there would invent a choice the model did not make.
            return pick if lo == 0 else [keep[0]]
        if pick[0] >= len(keep):
            return [keep[0]]
        return [keep[pick[0]]]
    return f
