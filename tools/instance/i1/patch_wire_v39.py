"""Wire v39 end to end, with the roles looked up rather than passed.

The failure mode this design avoids is the one rerank_deploy.sh warns about in capitals: the
prompt format lives in FOUR places (build_rerank, build_sft, lm/agent, rl_rollout) and if one of
them renders differently the model is scored on inputs it never trained on, silently, with no
error and no size change. Adding a `roles` argument would have created a fifth thing each caller
must remember.

So `serialize_stateless` resolves the roles itself from `deck_name`, via a cached tuning.json
read -- lm/vocab._fleet_names already reads that file the same way, and
build_rerank_submission.py ships it in the bundle. Callers that already pass `deck_name` (all of
them, since the ID segment needs it) get roles for free.

build_sft and rl_rollout need no change at all: both build their serializer from
`dict(rl_config.PROMPT_FMT)`, so updating that one dict propagates.
"""
import os

# ---------------------------------------------------------------- 1. roles lookup by deck name
P = os.path.join(os.getcwd(), "lm/roles.py")
s = open(P).read()
if "def for_deck" not in s:
    s += '''

_TUNING = None


def for_deck(name):
    """Roles for a deck NAME, read from agents/tuning.json (cached).

    Looked up rather than passed so a caller cannot forget it: the prompt format already lives
    in build_rerank, build_sft, lm/agent and rl_rollout, and a mismatch between any two of them
    is silent. tuning.json ships in the submission bundle.
    """
    global _TUNING
    if not name:
        return {}
    if _TUNING is None:
        import json
        import os as _os
        root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        try:
            _TUNING = json.load(open(_os.path.join(root, "agents", "tuning.json")))
        except Exception:
            _TUNING = {}
    return resolve(_TUNING.get(name) or {})
'''
    open(P, "w").write(s)
    print("lm/roles.py: added for_deck()")

# ---------------------------------------------------------------- 2. serialize resolves them
P = os.path.join(os.getcwd(), "lm/serialize.py")
s = open(P).read()
OLD = '    mine = render_my_deck(deck_ids, obs, deck_mode, deck_shuffle, roles)'
NEW = ('    if roles is None and deck_mode == "roles":\n'
       '        from lm.roles import for_deck\n'
       '        roles = for_deck(deck_name)\n'
       '    mine = render_my_deck(deck_ids, obs, deck_mode, deck_shuffle, roles)')
if 'from lm.roles import for_deck' not in s:
    assert s.count(OLD) == 1, "render_my_deck call anchor"
    s = s.replace(OLD, NEW)
    open(P, "w").write(s)
    print("lm/serialize.py: roles resolved from deck_name")

# ---------------------------------------------------------------- 3. make_lm_agent passes fmt
P = os.path.join(os.getcwd(), "lm/agent.py")
s = open(P).read()
if "board_facts" not in s:
    OLD_SIG = ('def make_lm_agent(deck, profile=None, model=None, glossary="full", deck_name=None,\n'
               '                  deck_glossary=True, deck_mode="static", deck_shuffle=False):')
    NEW_SIG = ('def make_lm_agent(deck, profile=None, model=None, glossary="full", deck_name=None,\n'
               '                  deck_glossary=True, deck_mode="static", deck_shuffle=False,\n'
               '                  board_facts=False, identify="both"):')
    assert s.count(OLD_SIG) == 1, "make_lm_agent signature"
    s = s.replace(OLD_SIG, NEW_SIG)
    OLD_SER = ('    _ser = lambda o: serialize_stateless(o, deck_ids=deck_ids, glossary=glossary,  # noqa: E731\n'
               '                                         deck_name=deck_name, deck_mode=deck_mode,\n'
               '                                         deck_shuffle=deck_shuffle)')
    NEW_SER = ('    _ser = lambda o: serialize_stateless(o, deck_ids=deck_ids, glossary=glossary,  # noqa: E731\n'
               '                                         deck_name=deck_name, deck_mode=deck_mode,\n'
               '                                         deck_shuffle=deck_shuffle,\n'
               '                                         board_facts=board_facts, identify=identify)')
    assert s.count(OLD_SER) == 1, "make_lm_agent serializer"
    s = s.replace(OLD_SER, NEW_SER)
    open(P, "w").write(s)
    print("lm/agent.py: board_facts/identify threaded")

# ---------------------------------------------------------------- 4. PROMPT_FMT -> v39
P = os.path.join(os.getcwd(), "tools/rl_config.py")
s = open(P).read()
OLD = 'PROMPT_FMT = dict(glossary="none", deck_mode="remaining", deck_shuffle=True)'
NEW = ('# v39 (2026-07-31). deck_mode="roles" groups DECK by prompt_roles with ids sorted inside a\n'
       '# group, which removes the file-order fingerprint structurally, so deck_shuffle is off.\n'
       '# board_facts adds `need:N` (type-aware energies short of any damaging attack) and `rt:N`\n'
       '# on both sides: measured, the v37 prompt contained attack costs 0 times and retreat costs\n'
       '# 0 times in 40,000 samples, and `attach` (20.3% of ceiling) and `retreat` (33.6%, chance\n'
       '# 31.2%) are exactly the two kinds whose decisive input was therefore missing.\n'
       '# identify="op" drops `ID ME`, redundant given DECK[].\n'
       'PROMPT_FMT = dict(glossary="none", deck_mode="roles", deck_shuffle=False,\n'
       '                  board_facts=True, identify="op")\n'
       'PROMPT_FMT_V37 = dict(glossary="none", deck_mode="remaining", deck_shuffle=True)')
if 'PROMPT_FMT_V37' not in s:
    assert s.count(OLD) == 1, "PROMPT_FMT anchor"
    s = s.replace(OLD, NEW)
    open(P, "w").write(s)
    print("rl_config.py: PROMPT_FMT -> v39 (v37 kept as PROMPT_FMT_V37)")
