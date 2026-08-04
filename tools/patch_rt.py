#!/usr/bin/env python3
"""Rewrite `rt:N` in ALREADY-RENDERED pool rows, from the card's printed cost to the live one.

The base pools are 14.3M rows of rendered text; the raw observations are gone (pool_daemon.sh
deletes each selfplay tag once converted), so they cannot be re-rendered, and the tag composition
was never recorded so rebuilding from scratch would change more than the bug. But the rendered
state already carries everything the live cost depends on -- card ids, attached energy, tools, HP,
the stadium, and both boards -- so it can be recomputed in place, changing nothing else.

It does NOT re-implement the cost rules. It parses the text back into the minimal shape
`lm.costs.effective_retreat_cost` reads and calls that, so there is exactly one implementation and
no way for the two to drift.

VALIDATION IS IDEMPOTENCE. Freshly rendered text already has the correct `rt`, so running this on
it must change nothing. `--validate` renders real prompts and asserts exactly that; any parse
error shows up as a diff. Run it before touching a pool.

    python3 tools/patch_rt.py --validate --games 20
    python3 tools/patch_rt.py --inp data/rerank/v40_base.jsonl.gz --out /tmp/patched.jsonl.gz
"""

import argparse
import gzip
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "cg-lib")):
    if p not in sys.path:
        sys.path.insert(0, p)

# `cN[*]:hp/maxHp[|energy][|tools][ need:N][ rt:N]` -- see lm.serialize._pk
_MON = re.compile(
    r"c(?P<cid>\d+)\*?:(?P<hp>\d+)/(?P<mx>\d+)"
    r"(?P<segs>(?:\|[^\s,\]]+)*)"
    r"(?P<facts>(?: need:\d+)?(?: rt:(?P<rt>\d+))?)")
_SIDE = re.compile(r" ME (?P<me>.*?) \| OP (?P<op>.*?)(?= ID | \|\| |$)", re.S)
_STAD = re.compile(r"stad:c(\d+)")
_ENERGY_SEG = re.compile(r"^[A-Z*][A-Z0-9*]*$")
# Active and bench are read from their OWN brackets. Taking "the first _MON match is the Active"
# breaks the moment a side has none: the engine renders `A[-]`, which matches nothing, so every
# bench slot shifted up one and a bench Gravity Gemstone was read as an Active one (caught by
# --validate on chandelure, +1 retreat cost on both sides).
_A_GRP = re.compile(r"A\[(?P<v>[^\]]*)\]")
_B_GRP = re.compile(r"B\[(?P<v>[^\]]*)\]")


def _letters_to_types():
    from lm import vocab
    return {v: k for k, v in vocab._ENERGY_LETTER.items()}


_L2T = None


def _parse_energy(seg):
    """`G2C` -> [1, 1, 0]. Counts are per DISTINCT letter, per lm.serialize._energy_counts."""
    global _L2T
    if _L2T is None:
        _L2T = _letters_to_types()
    out = []
    for letter, num in re.findall(r"(TR|\*|[A-Z])(\d*)", seg):
        t = _L2T.get(letter)
        if t is None:
            continue
        out += [t] * (int(num) if num else 1)
    return out


def _mon_from_match(m):
    energies, tools = [], []
    for seg in [s for s in m.group("segs").split("|") if s]:
        if seg.startswith("c") and seg[1:].split(",")[0].isdigit():
            tools = [{"id": int(x[1:])} for x in seg.split(",") if x.startswith("c")]
        elif _ENERGY_SEG.match(seg):
            energies = _parse_energy(seg)
    return {"id": int(m.group("cid")), "hp": int(m.group("hp")),
            "maxHp": int(m.group("mx")), "energies": energies, "tools": tools}


def patch_state(state):
    """Return `state` with every `rt:N` replaced by the live cost. Unchanged if there is none."""
    from lm.costs import effective_retreat_cost
    sides = _SIDE.search(state)
    if not sides:
        return state
    stad = _STAD.search(state)
    # A side's own Pokemon in board order: Active then bench, exactly as _side renders them.
    parsed = {}
    for key in ("me", "op"):
        text, base = sides.group(key), sides.start(key)
        got = {"active": [], "bench": []}
        for slot, rx in (("active", _A_GRP), ("bench", _B_GRP)):
            g = rx.search(text)
            if not g:
                continue
            off = base + g.start("v")
            for m in _MON.finditer(g.group("v")):
                got[slot].append((off + m.start(), m, _mon_from_match(m)))
        parsed[key] = got
    obs = {"current": {
        "players": [{"active": [x[2] for x in parsed[k]["active"]],
                     "bench": [x[2] for x in parsed[k]["bench"]]} for k in ("me", "op")],
        "stadium": [{"id": int(stad.group(1))}] if stad else []}}

    edits = []                      # (absolute span of the rt digits, replacement)
    for pi, key in ((0, "me"), (1, "op")):
        for slot in ("active", "bench"):
            for abs_start, m, mon in parsed[key][slot]:
                if m.group("rt") is None:
                    continue
                live = effective_retreat_cost(obs, pi, mon)
                if live is None:
                    continue
                s, e = m.span("rt")
                edits.append((abs_start - m.start() + s, abs_start - m.start() + e, str(live)))
    if not edits:
        return state
    out, last = [], 0
    for s, e, rep in sorted(edits):
        out.append(state[last:s])
        out.append(rep)
        last = e
    out.append(state[last:])
    return "".join(out)


def validate(games, decks):
    import library
    from lm.agent import make_lm_agent
    from lm.serialize import serialize_stateless
    from tools import rl_config
    from tools.mirror_env import MirrorEngine
    fmt = dict(rl_config.PROMPT_FMT)
    eng = MirrorEngine()
    tuning = json.load(open(os.path.join(ROOT, "agents", "tuning.json")))
    names = [d.strip() for d in decks.split(",") if d.strip()] or sorted(library.list_decks())
    n = bad = 0
    first = None
    for di, dk in enumerate(names):
        ids = [int(x) for x in open(library.deck_path(dk)) if x.strip()]
        agent = make_lm_agent(ids, tuning.get(dk, {}), model=None)
        for g in range(games):
            obs = eng.start(ids, list(ids), 100000 + di * 1000 + g, mirror=0)
            if obs is None:
                continue
            try:
                for _ in range(4000):
                    cur = obs.get("current") or {}
                    if cur.get("result", -1) != -1 or not obs.get("select"):
                        break
                    st = serialize_stateless(obs, deck_ids=ids, deck_name=dk, **fmt)
                    n += 1
                    got = patch_state(st)
                    if got != st:
                        bad += 1
                        if first is None:
                            first = (dk, st, got)
                    obs = eng.select(agent(obs))
            except Exception:
                pass
            finally:
                eng.finish()
        print("  %-24s %7d checked, %d changed" % (dk, n, bad), flush=True)
    print("\nIDEMPOTENCE on freshly rendered text: %d prompts, %d changed" % (n, bad))
    if first:
        dk, a, b = first
        print("  first disagreement (%s):" % dk)
        for i in range(min(len(a), len(b))):
            if a[i] != b[i]:
                print("    rendered: ...%s..." % a[max(0, i - 70):i + 40])
                print("    patched : ...%s..." % b[max(0, i - 70):i + 40])
                break
    return bad == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inp")
    ap.add_argument("--out")
    ap.add_argument("--field", default="state")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--decks", default="")
    a = ap.parse_args()

    if a.validate:
        sys.exit(0 if validate(a.games, a.decks) else 1)
    if not (a.inp and a.out):
        sys.exit("need --inp and --out (or --validate)")

    n = changed = 0
    with gzip.open(a.inp, "rt") as f, gzip.open(a.out, "wt") as o:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                o.write(line)
                continue
            s = r.get(a.field)
            if isinstance(s, str):
                p = patch_state(s)
                if p != s:
                    r[a.field] = p
                    changed += 1
            o.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
            if n % 500000 == 0:
                print("  %d rows, %d changed" % (n, changed), flush=True)
    print("%d rows, %d changed (%.1f%%)" % (n, changed, 100.0 * changed / max(1, n)))


if __name__ == "__main__":
    main()
