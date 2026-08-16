"""v39 wiring: deck_mode='roles', identify='op', and board_facts threaded to the state renderer.

Everything is an explicit parameter of serialize_stateless so v37 stays byte-reproducible, and
`rl_config.PROMPT_FMT` remains the single value all three consumers read -- build_rerank,
lm/agent and rl_rollout.make_serializer must agree or the model is scored on inputs it never
trained on, silently (rerank_deploy.sh says exactly this).
"""
import os
import re

P = os.path.join(os.getcwd(), "lm/serialize.py")
s = open(P).read()
if "roles" in s.split("DECK_MODES =")[1][:40]:
    print("already patched")
    raise SystemExit(0)

# 1) new deck mode
s = s.replace('DECK_MODES = ("static", "remaining")',
              'DECK_MODES = ("static", "remaining", "roles")')

# 2) render_my_deck: roles mode
OLD = '''    if not shuffle:
        return "DECK[" + _tok_multiset(ids) + "]"'''
NEW = '''    if mode == "roles":
        # Groups in a FIXED order so a card's ROLE is readable from its POSITION, ids sorted
        # WITHIN a group so the order is a function of the CONTENTS and cannot fingerprint the
        # deck. Shuffling is therefore neither needed nor applied. Empty groups are omitted --
        # each group carries its own marker, so position need not be reserved, and "no win
        # cards left in the library" is itself information.
        from lm.roles import group as _group, UNLABELLED as _UNL
        mark = {"win": "win", "engine": "eng", "line": "line", "fuel": "fuel",
                "tech": "tech", "filler": "fil", _UNL: "oth"}
        cnt = Counter(ids)
        parts = []
        for r, g in _group(sorted(cnt), roles or {}):
            body = ",".join(vocab.card_tok(k) + (("x%d" % cnt[k]) if cnt[k] > 1 else "")
                            for k in g)
            parts.append("%s[%s]" % (mark.get(r, r), body))
        return "DECK " + " ".join(parts)
    if not shuffle:
        return "DECK[" + _tok_multiset(ids) + "]"'''
assert s.count(OLD) == 1, "render_my_deck anchor"
s = s.replace(OLD, NEW)
s = s.replace('def render_my_deck(deck_ids, obs=None, mode="static", shuffle=False):',
              'def render_my_deck(deck_ids, obs=None, mode="static", shuffle=False, roles=None):')
# roles mode with an empty remaining library must still say so
s = s.replace('            return "DECK[]"          # empty is INFORMATION',
              '            return ("DECK" if mode == "roles" else "DECK[]")'
              '          # empty is INFORMATION')

# 3) serialize_stateless signature + plumbing
s = s.replace('''def serialize_stateless(obs, deck_ids=None, glossary="full", deck_name=None,
                        deck_mode="static", deck_shuffle=False):''',
              '''def serialize_stateless(obs, deck_ids=None, glossary="full", deck_name=None,
                        deck_mode="static", deck_shuffle=False, roles=None,
                        board_facts=False, identify="both"):''')
s = s.replace('    mine = render_my_deck(deck_ids, obs, deck_mode, deck_shuffle)\n'
              '    return (head + (mine + " " if mine else "") + render_state(obs, deck_name)\n'
              '            + " || " + render_options(obs))',
              '    mine = render_my_deck(deck_ids, obs, deck_mode, deck_shuffle, roles)\n'
              '    return (head + (mine + " " if mine else "")\n'
              '            + render_state(obs, deck_name, board_facts=board_facts,\n'
              '                           identify=identify)\n'
              '            + " || " + render_options(obs))')

# 4) render_state gains the two knobs and passes board_facts to both sides
m = re.search(r"def render_state\(obs(?:, ([^)]*))?\):", s)
assert m, "render_state signature"
s = s[:m.start()] + ("def render_state(obs, deck_name=None, board_facts=False, "
                     "identify=\"both\"):") + s[m.end():]
s = re.sub(r"_side\(([^,]+), True\)", r"_side(\1, True, board_facts)", s)
s = re.sub(r"_side\(([^,]+), False\)", r"_side(\1, False, board_facts)", s)

open(P, "w").write(s)
print("patched deck_mode=roles + identify + board_facts plumbing")
