"""Build the dragapult_dusknoir pilot exactly as the shipped submission builds it.

ONE place decides prompt format + plan wrapper + fallback, shared by local play
(agents/lm_dusknoir.py, i.e. a human opponent) and measurement
(tools/kenkyu/battle_eval.py). When those two disagree the研究 measures a pilot nobody
plays against, and the disagreement is silent -- the model simply receives inputs it has
never seen and the win rate quietly drops. That exact bug shipped twice during the
competition, which is why this module exists rather than two call sites with the same
keyword arguments typed out.

The values are copied from the final submission bundle (dusk_v4 main.py, model fld_r49b).
"""
import os

# dusk_plan gates SIX rules on environment flags read at IMPORT time. A wrapper naming a
# gated-off rule does not raise -- the rule just returns an empty set and constrains nothing,
# which is how a live submission ran with two of its five wrapper rules inert. Set before
# anything imports dusk_plan.
for _v in ("DUSK_NEW_RULES", "DUSK_CLOPS_HOLD", "DUSK_FRONT_DIVE", "DUSK_BOSS_LETHAL",
           "DUSK_TIPS", "DUSK_SPIKE"):
    os.environ.setdefault(_v, "1")

DECK_NAME = "dragapult_dusknoir"
# rl_config.DUSK_FMT, inlined so this module does not drag the RL research tree in.
PROMPT_FMT = {"glossary": "none", "deck_mode": "none", "deck_shuffle": False,
              "board_facts": True, "identify": "op", "menu_dedup": True, "hidden_facts": True}
WRAP_RULES = ("lethal_now", "clops_hold", "judge_timing", "spare_ex_bench", "retreat_energy",
              "stadium_replace", "search_bottom", "setup_search", "front_dive", "promote_dive",
              "promote_line", "lethal_boss", "candy_line", "noir_critical", "stadium_bump",
              "hammer_now", "spike_candy", "spike_race", "hammer_spare", "lethal_line",
              "draw_cap")


def make_pilot(model=None, deck_name=DECK_NAME, wrap=True, profile=None):
    """``agent(obs) -> list[int]`` for the deck, piloted by ``model`` (None = pure engine_v2).

    ``wrap`` applies the hand-authored plan rules on top: they take whole decision families
    (lethal lines, energy denial timing, Stadium replacement) away from the model, and they
    are part of what was measured on the ladder -- turning them off measures a different
    pilot, not a cleaner one."""
    import json
    import library
    from lm.agent import make_lm_agent

    deck = library.read_deck(deck_name)
    if profile is None:
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "agents", "tuning.json")) as f:
            profile = json.load(f).get(deck_name, {})
    agent = make_lm_agent(deck, profile, model=model, deck_name=deck_name, **PROMPT_FMT)
    if wrap and model is not None:
        # strict=False == the shipped "planfilter:" head: the rules NARROW the menu and the
        # model still ranks inside what survives. strict=True would let the rule pick outright,
        # which is a different (unmeasured) pilot.
        from lm.plan_filter import make_plan_rule
        agent = make_plan_rule(agent, list(WRAP_RULES), strict=False)
    return agent
