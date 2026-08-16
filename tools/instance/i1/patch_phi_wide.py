"""Widen the setup potential from "can it pay" to the whole template.

The narrow version moved on 19 of 281 pairs (6.8%), because it only changes when energy lands on
a line body -- and attach is 16.5% of branch points while play is 45.9% and card 20.4%.  The
interventions that actually won (+6.55 encoder, +10.62 4B) moved Dreepy 1.15 -> 1.69 and Drakloak
0.40 -> 0.72, which are play and card decisions.  A potential that cannot see them is describing
the wrong thing.

So Phi becomes the human template, each part capped at what the guides ask for and nothing more:

    0.40  line bodies, up to 2       "bench Dreepy x3, two at minimum"
    0.30  a Drakloak (or Dragapult)  "evolve on turn 2 and draw with Recon Directive"
    0.30  energy toward {R}{P}       "energy is extremely tight -- attach early and often"

Capped, so it cannot be farmed: a fourth Dreepy scores the same as the second, and a third energy
the same as the second.  Uncapped it would reward over-benching, and bench-outs are already 26%
of this deck's losses ([[bench-out-losses-are-decklists]]), and over-attaching, which feeds the
opponent attack that counts energy on BOTH actives.

Still zero estimation error -- every term is read off the board, not off a playout.
"""
import os

p = "/root/ptcg/repo/tools/rl_branch.py"
s = open(p).read()

old = '''def _setup_potential(state, pilot_i):
    """How close our best line body is to paying Phantom Dive, in [0, 1].

    Only {R} and {P} count: a {D} on a Dreepy pays nothing toward the attack, and rewarding it
    would teach the attachment that [[weak-deck-bottleneck-fixes]] already had to correct for.
    """
    ob = state.get("observation") or {}
    cur = ob.get("current") or {}
    pls = cur.get("players") or []
    if pilot_i >= len(pls):
        return 0.0
    me = pls[pilot_i] or {}
    bodies = [x for x in ([(me.get("active") or [None])[0]] + list(me.get("bench") or []))
              if isinstance(x, dict)]
    best = 0.0
    for b in bodies:
        if b.get("id") not in _PD_LINE:
            continue
        useful = sum(1 for e in _energy_ids(b) if e in _PD_COST)
        best = max(best, min(1.0, useful / 2.0))
    return best'''
new = '''_W_BODIES, _W_EVOLVE, _W_ENERGY = 0.40, 0.30, 0.30
_TARGET_BODIES = 2           # "Dreepy x3, two at minimum" -- the minimum is what gets rewarded


def _setup_potential(state, pilot_i):
    """How far along the human opening we are, in [0, 1].

    Three capped terms, each one a thing the guides name: bodies on the line, the Drakloak that
    turns into both the draw engine and the attacker, and energy that actually pays {R}{P}.
    Only {R} and {P} count -- a {D} on a Dreepy pays nothing toward the attack, and rewarding it
    would teach the attachment that [[weak-deck-bottleneck-fixes]] already had to correct for.
    """
    ob = state.get("observation") or {}
    cur = ob.get("current") or {}
    pls = cur.get("players") or []
    if pilot_i >= len(pls):
        return 0.0
    me = pls[pilot_i] or {}
    bodies = [x for x in ([(me.get("active") or [None])[0]] + list(me.get("bench") or []))
              if isinstance(x, dict)]
    line = [b for b in bodies if b.get("id") in _PD_LINE]
    n_line = min(len(line), _TARGET_BODIES)
    evolved = 1.0 if any(b.get("id") in (120, 121) for b in line) else 0.0
    best_e = 0.0
    for b in line:
        useful = sum(1 for e in _energy_ids(b) if e in _PD_COST)
        best_e = max(best_e, min(1.0, useful / 2.0))
    return (_W_BODIES * (n_line / _TARGET_BODIES)
            + _W_EVOLVE * evolved
            + _W_ENERGY * best_e)'''
assert s.count(old) == 1, "potential anchor"
s = s.replace(old, new)
open(p + ".new", "w").write(s)
os.replace(p + ".new", p)
print("patched")

import subprocess
r = subprocess.run(["python3", "-c", """
import sys
sys.path.insert(0, "/root/ptcg/repo/tools"); sys.path.insert(0, "/root/ptcg/repo")
import rl_branch as R
mk = lambda bodies: {"observation": {"current": {"players": [
    {"active": [bodies[0]] if bodies else [], "bench": bodies[1:]}, {}]}}}
E = lambda *ids: {"energy": [{"id": i} for i in ids]}
def b(cid, *e): return dict({"id": cid}, **(E(*e) if e else {"energy": []}))
cases = [
    ("empty board",                     []),
    ("1 Dreepy",                        [b(119)]),
    ("2 Dreepy",                        [b(119), b(119)]),
    ("3 Dreepy (capped at 2)",          [b(119), b(119), b(119)]),
    ("2 Dreepy + Drakloak",             [b(120), b(119), b(119)]),
    ("2 Dreepy + Drakloak {R}",         [b(120, 2), b(119), b(119)]),
    ("2 Dreepy + Drakloak {R}{P}",      [b(120, 2, 5), b(119), b(119)]),
    ("... + a third energy",            [b(120, 2, 5, 5), b(119), b(119)]),
    ("2 Dreepy + Drakloak {D}{D}",      [b(120, 7, 7), b(119), b(119)]),
    ("2 Munkidori (off line)",          [b(112), b(112)]),
]
for lbl, bd in cases:
    print("  %-30s %.3f" % (lbl, R._setup_potential(mk(bd), 0)))
"""], capture_output=True, text=True)
print(r.stdout or r.stderr[-600:])
