"""Make build_sft render the CURRENT prompt format, and pass the deck NAME.

`tools/_legacy_decoder/build_sft.py` calls `serialize_stateless(obs, deck_ids=...)` bare at four
sites. Bare means the LEGACY format: full-rules glossary (~838 tokens), no `DECK[...]`, no
`ID ME d_<deck> a_<arch>`. The reranker and the RL loop use
`rl_config.PROMPT_FMT = dict(glossary="none", deck_mode="remaining", deck_shuffle=True)` plus the
deck ids AND the deck name -- that is what `rl_rollout.make_serializer` exists to guarantee.

Two reasons this must be fixed before regenerating teacher data:
  * distillation compares a teacher and a student on the SAME state text; different prompts make
    the transfer meaningless;
  * the legacy prompt is ~3.4x longer, which multiplies teacher training and rollout cost.

The deck name is available: gen_selfplay writes `header["agents"] = {"0": nameA, "1": nameB}`
(gen_selfplay.py:390, where `order = (nameA, nameB)`).
"""
import os

P = os.path.join(os.getcwd(), "tools/_legacy_decoder/build_sft.py")
s = open(P).read()

if "_ser_cur" in s:
    print("already patched")
    raise SystemExit(0)

# 1) helper, inserted right after the serialize import line
IMP = ("from lm.serialize import serialize_stateless, render_logs, multipick_substate, "
       "STOP  # noqa: E402\n")
assert s.count(IMP) == 1, "import anchor not unique"
HELPER = IMP + '''
import rl_config  # noqa: E402


def _ser_cur(obs, deck_ids=None, deck_name=None):
    """serialize_stateless in the CURRENT prompt format.

    Bare serialize_stateless() renders the legacy full-rules glossary with no DECK[] and no
    `ID ME` -- a ~838-token prompt for models trained on ~245. rl_config.PROMPT_FMT is the single
    source of truth and the deck NAME is not optional (it renders `ID ME d_x a_y`)."""
    return serialize_stateless(obs, deck_ids=deck_ids, deck_name=deck_name,
                               **dict(rl_config.PROMPT_FMT))
'''
s = s.replace(IMP, HELPER)

# 2) _multipick_pairs takes the deck name and uses the helper
OLD_MP_SIG = "def _multipick_pairs(step, deck_ids"
assert OLD_MP_SIG in s, "multipick signature not found"
import re
m = re.search(r"def _multipick_pairs\(step, deck_ids[^)]*\):", s)
assert m, "multipick signature regex"
s = s[:m.start()] + "def _multipick_pairs(step, deck_ids, deck_name=None):" + s[m.end():]

for old, new in (
    ('pairs.append((serialize_stateless(sub, deck_ids=deck_ids), encode_option(opts[pos_i], obs)))',
     'pairs.append((_ser_cur(sub, deck_ids, deck_name), encode_option(opts[pos_i], obs)))'),
    ('pairs.append((serialize_stateless(sub, deck_ids=deck_ids), STOP))',
     'pairs.append((_ser_cur(sub, deck_ids, deck_name), STOP))'),
    ('pairs = [(serialize_stateless(s["obs"], deck_ids=game_decks.get(p)), action)]',
     'pairs = [(_ser_cur(s["obs"], game_decks.get(p), _dname(header, p)), action)]'),
    ('pairs = [(serialize_stateless(s["obs"], deck_ids=game_decks.get(p)), _executed_target(s))]',
     'pairs = [(_ser_cur(s["obs"], game_decks.get(p), _dname(header, p)), _executed_target(s))]'),
    ('pairs = _multipick_pairs(s, game_decks.get(p))',
     'pairs = _multipick_pairs(s, game_decks.get(p), _dname(header, p))'),
):
    assert s.count(old) == 1, "call-site anchor not unique: %s" % old[:60]
    s = s.replace(old, new)

# 3) the deck-name accessor
DN = '''

def _dname(header, p):
    """Deck NAME for player p. gen_selfplay writes header["agents"] = {"0": nameA, "1": nameB}."""
    a = (header or {}).get("agents") or {}
    return a.get(str(p)) or a.get(p)
'''
anchor = "\ndef _multipick_pairs("
assert s.count(anchor) == 1
s = s.replace(anchor, DN + anchor)

open(P, "w").write(s)
print("patched", P)
