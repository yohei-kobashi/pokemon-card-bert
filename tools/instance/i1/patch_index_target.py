"""build_sft: add --target-mode index, emitting the MENU INDEX instead of the action string.

Measured with the Qwen3.5 tokenizer on the freshly built data: the action-string target
(`card:c743@DECK0`) is 7 tokens at p50 and up to 10, while the prompt is 449 at p50. Two
consequences for a teacher whose whole purpose is to hand a candidate distribution to the
cross-encoder:

  * RL rollouts pay 7-10 decode steps per decision instead of 1;
  * the clean distillation trick -- ONE forward gives a distribution over ALL candidates by
    restricting logits to the answer tokens -- only works if the answer is one token.

The prompt already numbers the menu (`:: 0=play:c1086 1=play:c1264 2=attach:c18@ACTIVE0 ...`),
so the index IS the answer. Verified on 3,088,497 records of the new data before doing this:
100% of records are `n1-1` (exactly-one selects), 0 STOP targets, 0 comma-joined multi-targets.
So every record has exactly one chosen option that appears in the rendered menu, and the index is
unambiguous. Option counts: p50 5, p90 13, p99 26, max 51, so the target is 1-2 tokens.

Indices are taken from the recorded option positions, never by string-matching the target against
the menu -- duplicate options are common (two copies of a card in hand render identically) and
matching would pick an arbitrary one.
"""
import os
import re

P = os.path.join(os.getcwd(), "tools/_legacy_decoder/build_sft.py")
s = open(P).read()

if "TARGET_MODE" in s:
    print("already patched")
    raise SystemExit(0)

# 1) module global + helper, after _executed_indices
ANCHOR = '''def _executed_indices(step):
    """Original option indices the agent picked, in recorded order."""
    return (step.get("executed") if step.get("explored") else step.get("action")) or []
'''
assert s.count(ANCHOR) == 1, "indices anchor"
NEW = ANCHOR + '''

TARGET_MODE = "action"          # "action" = encode_option string, "index" = menu position


def _idx_target(step, remap=None):
    """The chosen option's position in the RENDERED menu, as a string.

    `remap` is the substate's remaining-original-index list for a multi-pick step, because the
    substate menu is a subset and renumbers. Returns None when the position cannot be resolved,
    so the caller can drop the record rather than emit a wrong label."""
    idx = _executed_indices(step)
    if len(idx) != 1:
        return None
    j = idx[0]
    if remap is not None:
        try:
            j = list(remap).index(j)
        except ValueError:
            return None
    return str(j)
'''
s = s.replace(ANCHOR, NEW)

# 2) the two direct call sites
for old, new in (
    ('pairs = [(_ser_cur(s["obs"], game_decks.get(p), _dname(header, p)), action)]',
     'pairs = [(_ser_cur(s["obs"], game_decks.get(p), _dname(header, p)),\n'
     '                                  (_idx_target(s) if TARGET_MODE == "index" else action))]'),
    ('pairs = [(_ser_cur(s["obs"], game_decks.get(p), _dname(header, p)), _executed_target(s))]',
     'pairs = [(_ser_cur(s["obs"], game_decks.get(p), _dname(header, p)),\n'
     '                                      (_idx_target(s) if TARGET_MODE == "index"\n'
     '                                       else _executed_target(s)))]'),
):
    assert s.count(old) == 1, "call site: %s" % old[:50]
    s = s.replace(old, new)

# 3) multipick: index is relative to the substate menu
OLD_MP = ('        pairs.append((_ser_cur(sub, deck_ids, deck_name), '
          'encode_option(opts[pos_i], obs)))')
NEW_MP = ('        _t = (str(_remaining.index(pos_i)) if TARGET_MODE == "index"\n'
          '              else encode_option(opts[pos_i], obs))\n'
          '        pairs.append((_ser_cur(sub, deck_ids, deck_name), _t))')
assert s.count(OLD_MP) == 1, "multipick append"
s = s.replace(OLD_MP, NEW_MP)
# the substate call must keep `remaining`
OLD_SUB = "        sub, _remaining, _stop = multipick_substate(obs, picked)"
if OLD_SUB not in s:
    m = re.search(r"        sub, (\w+), (\w+) = multipick_substate\(obs, picked\)", s)
    assert m, "multipick_substate call"
    s = s[:m.start()] + OLD_SUB + s[m.end():]

# 4) drop records whose index could not be resolved
OLD_EMIT = "                    for prompt_body, act_target in pairs:"
NEW_EMIT = ("                    pairs = [(a, b) for a, b in pairs if b is not None]\n"
            + OLD_EMIT)
assert s.count(OLD_EMIT) == 1, "emit loop"
s = s.replace(OLD_EMIT, NEW_EMIT)

# 5) the CLI flag
OLD_ARG = '    ap.add_argument("--modes", default="act,reason", help="comma list: act, reason")'
NEW_ARG = OLD_ARG + '''
    ap.add_argument("--target-mode", choices=["action", "index"], default="action",
                    help="'action' emits encode_option(chosen); 'index' emits the chosen "
                         "option's position in the rendered menu (1-2 tokens, and one forward "
                         "then yields a distribution over ALL candidates -- what distillation "
                         "into the cross-encoder needs)")'''
assert s.count(OLD_ARG) == 1, "modes arg"
s = s.replace(OLD_ARG, NEW_ARG)

# 6) set the global from args -- put it right after the parse
m = re.search(r"\n(\s*)args = ap\.parse_args\(\)\n", s)
assert m, "parse_args"
ind = m.group(1)
s = s[:m.end()] + "%sglobal TARGET_MODE\n%sTARGET_MODE = args.target_mode\n" % (ind, ind) + s[m.end():]

open(P, "w").write(s)
print("patched", P)
