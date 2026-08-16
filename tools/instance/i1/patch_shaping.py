"""Add a setup potential to the branch reward -- an OBSERVED board fact, not a playout estimate.

WHY A SHAPING TERM AND NOT A RULE.  Forcing the setup search was worth +6.55 (encoder) and
+10.62 (4B); forcing the energy rules was worth -2.25.  The difference is context: what to fetch
from the deck competes with nothing else on that menu, while where to attach depends on who is
going to attack, whether Munkidori needs its {D}, and whether the body is about to retreat.  A
rule can carry the first kind of knowledge and cannot carry the second.

WHY IT SHOULD HELP AT ALL.  Priced over 4,370 pairs from rounds 20/22/23:

    decision   pairs   share   mean|dQ|
    attach       722   16.5%     0.339     <- the most decidable frequent decision
    play        2006   45.9%     0.300
    card         892   20.4%     0.270
    attack       107    2.4%     0.312     <- and only 2.4% of branch points, which is why
                                              forcing phantom_dive measured +0.33 +- 0.87

With 24 playouts each Q has SE ~0.2, so |dQ| carries a ~0.22 noise floor and the TRUE signal is
about 0.12 for attach against 0.05 for card.  That is the real argument for shaping: it is not
about injecting my opinion, it is that "a body can pay {R}{P}" is a fact with zero estimation
error, replacing part of a statistic that is mostly noise.  [[rl-plateau-is-label-confidence]]
diagnosed the plateau as label confidence; this is the matching prescription.

WHERE IT IS EVALUATED.  On the state IMMEDIATELY after each candidate selection, never at the
terminal.  The rollout is engine_v2 on both sides, so a terminal-measured setup term would score
engine_v2's energy handling rather than the choice being labelled.  The root's potential is
common to all candidates in a scenario and cancels in qw-ql, so only the successor is needed.

CAPPED AT THE TEMPLATE.  Two useful energies is what Phantom Dive costs and the cap sits exactly
there, so the term cannot be farmed by stacking energy -- which would also feed Myriad Leaf
Shower, the opponent attack that counts energy on BOTH actives.

OFF BY DEFAULT (RL_SETUP_GAMMA=0), and inert for any deck without Dragapult ex, so every existing
caller -- instance2's brancher included -- stays bit-identical until someone opts in.
"""
import os

p = "/root/ptcg/repo/tools/rl_branch.py"
s = open(p).read()

old = 'PRIZE_GAMMA = float(os.environ.get("RL_PRIZE_GAMMA", "0") or 0)'
new = '''PRIZE_GAMMA = float(os.environ.get("RL_PRIZE_GAMMA", "0") or 0)

# Setup shaping: see the module note. Evaluated on the successor of each candidate, capped at
# what Phantom Dive actually costs, and applied only to the deck it describes.
SETUP_GAMMA = float(os.environ.get("RL_SETUP_GAMMA", "0") or 0)
_PD_COST = (2, 5)            # basic {R}, basic {P}
_PD_LINE = (119, 120, 121)   # Dreepy / Drakloak / Dragapult ex -- energy carries up the line
_PULT = 121


def _energy_ids(body):
    for k in ("energy", "attachedEnergy", "energies"):
        v = body.get(k)
        if isinstance(v, list):
            return [(e.get("id") if isinstance(e, dict) else e) for e in v]
    return []


def _setup_potential(state, pilot_i):
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
assert s.count(old) == 1, "gamma anchor"
s = s.replace(old, new)

old2 = """                v = _playout(step["state"], pilot_i, agent_me, agent_opp)
                if v is not None:
                    vals[i].append(v)"""
new2 = """                v = _playout(step["state"], pilot_i, agent_me, agent_opp)
                if v is not None:
                    if shape:
                        # the successor's potential; the root's is common to every candidate in
                        # this scenario and cancels in the pairwise comparison
                        v += SETUP_GAMMA * _setup_potential(step["state"], pilot_i)
                    vals[i].append(v)"""
assert s.count(old2) == 1, "playout anchor"
s = s.replace(old2, new2)

old3 = """    vals = [[] for _ in selections]"""
new3 = """    # Only the deck the potential describes. Every other deck keeps the old reward exactly.
    shape = bool(SETUP_GAMMA) and _PULT in (my_deck or ())
    vals = [[] for _ in selections]"""
assert s.count(old3) == 1, "vals anchor"
s = s.replace(old3, new3)

open(p + ".new", "w").write(s)
os.replace(p + ".new", p)
print("patched rl_branch.py")

# quick self-check on synthetic states
import subprocess
r = subprocess.run(["python3", "-c", """
import os, sys
os.environ["RL_SETUP_GAMMA"] = "0.10"
sys.path.insert(0, "/root/ptcg/repo/tools"); sys.path.insert(0, "/root/ptcg/repo")
import importlib, rl_branch; importlib.reload(rl_branch)
mk = lambda bodies: {"observation": {"current": {"players": [{"active": [bodies[0]] if bodies else [],
      "bench": bodies[1:]}, {}]}}}
f = rl_branch._setup_potential
print("empty            ", f(mk([]), 0))
print("Dreepy, no energy", f(mk([{"id": 119, "energy": []}]), 0))
print("Dreepy {R}       ", f(mk([{"id": 119, "energy": [{"id": 2}]}]), 0))
print("Dreepy {R}{P}    ", f(mk([{"id": 119, "energy": [{"id": 2}, {"id": 5}]}]), 0))
print("Dreepy {R}{P}{P} ", f(mk([{"id": 119, "energy": [{"id": 2}, {"id": 5}, {"id": 5}]}]), 0))
print("Dreepy {D}{D}    ", f(mk([{"id": 119, "energy": [{"id": 7}, {"id": 7}]}]), 0))
print("Munkidori {R}{P} ", f(mk([{"id": 112, "energy": [{"id": 2}, {"id": 5}]}]), 0))
print("gamma            ", rl_branch.SETUP_GAMMA)
"""], capture_output=True, text=True)
print(r.stdout or r.stderr[-800:])
