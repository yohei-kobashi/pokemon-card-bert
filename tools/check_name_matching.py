"""Guard against the silent apostrophe-mismatch bug class.

The card DB mixes BOTH apostrophes: 53 names carry U+2019 (’), the rest U+0027 ('), and
`evolvesFrom` does the same. Any code that matches a card by NAME with a hard-coded
literal can therefore miss — and the miss is SILENT: a rule that never fires looks
exactly like a rule whose condition is never true. That is the same failure mode as the
raw-`o.cardId` doctrine bugs (docs: the dead-doctrine audit), and it is why the rule is:

    MATCH ON CARD ID. Use names only for a card FAMILY that has no id list,
    and then only through agents.engine_v2._norm_name.

Run:  python tools/check_name_matching.py       (exit 1 on a problem)
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "cg-lib"))

from cg.api import CardType, all_card_data  # noqa: E402

from agents.engine_v2 import _norm_name  # noqa: E402

CARDS = list(all_card_data())
CURLY = "’"
STRAIGHT = "'"
fail = 0


def bad(msg):
    global fail
    fail += 1
    print("FAIL " + msg)


# 1. Every evolvesFrom must resolve to a real card name EXACTLY. If this ever starts
#    failing, tools/audit_dead_cards.py will invent "evolves from X, not in the deck"
#    findings out of thin air (verified clean when the audit was written).
names = {c.name for c in CARDS}
norm_names = {_norm_name(n) for n in names}
for c in CARDS:
    if c.cardType != CardType.POKEMON or not c.evolvesFrom:
        continue
    if c.evolvesFrom in names:
        continue
    if _norm_name(c.evolvesFrom) in norm_names:
        bad(f"evolvesFrom needs apostrophe normalization: {c.cardId} {c.name!r} "
            f"<- {c.evolvesFrom!r}")
    else:
        bad(f"evolvesFrom names a card that does not exist: {c.cardId} {c.name!r} "
            f"<- {c.evolvesFrom!r}")

# 2. No shipped engine code may match a card name against a hard-coded literal
#    containing an apostrophe unless it goes through _norm_name. (The first version of
#    this lint did NOT fire on the very bug it was written for -- it demanded a space
#    after "'s", which `"Team Rocket's" in c.name` does not have. A check that cannot
#    fail is the same silent-zero trap it is meant to catch, so it is verified against
#    the real bug below.)
NAME_REF = re.compile(r"""\.name\b|["']name["']\s*\]""")
APOS_LIT = re.compile(r"""(?P<q>["'])(?:(?!(?P=q)).)*['’](?:(?!(?P=q)).)*(?P=q)""")


def lint_line(s):
    """True = this line matches a card name against an apostrophe-bearing literal."""
    s = s.strip()
    if s.startswith("#") or "_norm_name" in s:
        return False
    return bool(NAME_REF.search(s) and APOS_LIT.search(s))


# self-test the lint on the exact line that shipped broken, and on its fixed form
_BROKEN = '''        return bool(c and "Team Rocket's" in (c.name or ""))'''
_FIXED = '''        return bool(c and "team rocket's" in _norm_name(c.name))'''
if not lint_line(_BROKEN):
    bad("the lint does not fire on the known-broken line — it would never catch anything")
if lint_line(_FIXED):
    bad("the lint fires on the correct _norm_name form — it would be ignored as noise")

def code_lines(path):
    """Yield (lineno, line) for real code only. Docstrings must be skipped: this file's
    own explanation of the bug QUOTES the broken pattern, and the lint flagged the
    documentation of the fix as the bug."""
    depth = None
    for i, line in enumerate(open(path, encoding="utf-8"), 1):
        s = line.strip()
        if depth is None:
            m = re.match(r'''^[rbfu]*("""|\'\'\')''', s)
            if m:
                q = m.group(1)
                # one-line docstring?
                if not (len(s) > len(m.group(0)) and s.endswith(q)):
                    depth = q
                continue
        else:
            if depth in s:
                depth = None
            continue
        yield i, line


for rel in ("agents/engine_v2.py", "agents/_engine.py"):
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        continue
    for i, line in code_lines(p):
        if lint_line(line):
            bad(f"{rel}:{i} name-matched against a literal without _norm_name:\n     {line.strip()}")

# 3. _norm_name must actually collapse the two forms.
if _norm_name(f"Team Rocket{CURLY}s Hypnotizer") != _norm_name(f"Team Rocket{STRAIGHT}s Hypnotizer"):
    bad("_norm_name does not collapse U+2019 and U+0027")

# 4. The specific card that motivated this: it must be recognized as a Team Rocket's card.
from agents.engine_v2 import RocketsMewtwoL2  # noqa: E402
hyp = next((c for c in CARDS if _norm_name(c.name) == "team rocket's hypnotizer"), None)
if hyp is None:
    bad("Team Rocket's Hypnotizer not found in the pool (test needs updating)")
elif not RocketsMewtwoL2._is_tr(hyp.cardId):
    bad(f"_is_tr misses {hyp.cardId} {hyp.name!r} — the apostrophe bug is back")

curly_names = sum(1 for c in CARDS if CURLY in (c.name or ""))
print(f"\nchecked {len(CARDS)} cards ({curly_names} names use the curly apostrophe)")
print("OK — no silent name-matching mismatches" if not fail else f"\n{fail} problem(s)")
sys.exit(1 if fail else 0)
