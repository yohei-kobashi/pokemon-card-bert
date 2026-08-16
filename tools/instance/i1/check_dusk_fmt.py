"""Does the DUSK_FMT renderer produce what the stripped pool actually contains?

The pool was edited TEXTUALLY (a regex removed the DECK segment) while inference renders from
scratch with deck_mode='none'. Nothing forces those two to agree, and a mismatch is silent --
the model just scores badly. So compare their shapes directly, and check that rendering the OLD
format and applying the pool's own regex reproduces the NEW rendering byte for byte.
"""
import gzip, json, os, re, sys

ROOT = "/root/ptcg/repo"
for p in (ROOT, os.path.join(ROOT, "cg-lib"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import rl_config
from lm.serialize import serialize_stateless
from tools.mirror_env import DEFAULT_SO, MirrorEngine, play, _engine_agent, _load_deck

STRIP = re.compile(r"^DECK .*?(?=T\d+\.)", re.S)

# ---- 1. what the training rows look like now -------------------------------------------
rows = []
with gzip.open(os.path.join(ROOT, "data/rerank/v41_dusk11.jsonl.gz"), "rt") as f:
    for i, line in enumerate(f):
        rows.append(json.loads(line))
        if i >= 999:
            break
bad = [r for r in rows if "DECK" in r["state"][:200]]
heads = sorted({re.match(r"^(\S+)", r["state"]).group(1)[:3] for r in rows})
print("POOL   %d rows | rows still carrying DECK: %d | first tokens: %s"
      % (len(rows), len(bad), heads[:6]))
print("POOL   sample head: %r" % rows[0]["state"][:80])

# ---- 2/3. render the same live decision both ways ---------------------------------------
ids, oids = _load_deck("dragapult_dusknoir"), _load_deck("marnie_grimmsnarl")
new_fmt, old_fmt = dict(rl_config.DUSK_FMT), dict(rl_config.PROMPT_FMT)
caught = []


def capture(inner):
    def f(obs):
        if len(caught) < 20 and ((obs.get("select") or {}).get("option")):
            caught.append(obs)
        return inner(obs)
    return f


eng = MirrorEngine(DEFAULT_SO)
play(eng, capture(_engine_agent(ids)), _engine_agent(oids), ids, oids, 7, mirror=1)

same = shrink = 0
for obs in caught:
    a = serialize_stateless(obs, deck_ids=ids, deck_name="dragapult_dusknoir", **old_fmt)
    b = serialize_stateless(obs, deck_ids=ids, deck_name="dragapult_dusknoir", **new_fmt)
    same += (STRIP.sub("", a, count=1) == b)
    shrink += len(a) - len(b)
print("RENDER %d decisions | old-minus-DECK == new on %d of them" % (len(caught), same))
if caught:
    a = serialize_stateless(caught[0], deck_ids=ids, deck_name="dragapult_dusknoir", **old_fmt)
    b = serialize_stateless(caught[0], deck_ids=ids, deck_name="dragapult_dusknoir", **new_fmt)
    print("RENDER old head: %r" % a[:80])
    print("RENDER new head: %r" % b[:80])
    print("RENDER new contains 'DECK': %s | mean chars saved %.0f (%.1f%%)"
          % ("DECK" in b, shrink / len(caught), 100.0 * shrink / max(1, sum(
              len(serialize_stateless(o, deck_ids=ids, deck_name="dragapult_dusknoir",
                                      **old_fmt)) for o in caught))))
