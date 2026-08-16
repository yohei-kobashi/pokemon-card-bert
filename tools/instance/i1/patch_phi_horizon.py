"""Measure the potential at the END OF OUR TURN, not at the successor of one menu choice.

The successor-level version separated 9.1% of pairs and no pair by more than 0.15, which says the
two candidates at a single decision leave almost the same board -- of course they do: "play
trainer A" against "play trainer B" moves no setup counter at all. Setup accumulates over a TURN.
Choosing to search a Dreepy only shows up once it has been benched, which is two or three menus
later in the same turn.

So the potential is now sampled inside the playout, at the first state where the turn has passed
to the opponent. That is still OUR choice's consequence -- everything between the branch and the
turn boundary is the same policy playing on -- and it is still a board fact with no estimation
error. What it is not is the terminal state, which would be engine_v2's setup rather than ours.

_playout returns (value, phi_at_turn_end); phi is None if the game ended inside our own turn,
which is rare and is treated as "no information" rather than as zero.
"""
import os

p = "/root/ptcg/repo/tools/rl_branch.py"
s = open(p).read()

old = '''def _playout(state, pilot_i, agent_me, agent_opp, max_steps=4000):
    """Drive a branch to a terminal result with engine_v2 on both sides.
    Returns +1 / -1 for the PILOT (plus the PRIZE_GAMMA margin term when enabled),
    or None if the branch did not resolve."""
    steps = 0
    while steps < max_steps:
        ob = state.get("observation") or {}
        cur = ob.get("current")
        if cur is None:
            return None'''
new = '''def _playout(state, pilot_i, agent_me, agent_opp, max_steps=4000, want_phi=False):
    """Drive a branch to a terminal result with engine_v2 on both sides.

    Returns +1 / -1 for the PILOT (plus the PRIZE_GAMMA margin term when enabled), or None if
    the branch did not resolve. With want_phi, returns (value, phi) where phi is the setup
    potential at the moment OUR turn ended -- see _setup_potential and the note on why the
    turn boundary rather than the successor or the terminal.
    """
    steps = 0
    phi = None
    def _ret(v):
        return (v, phi) if want_phi else v
    while steps < max_steps:
        ob = state.get("observation") or {}
        cur = ob.get("current")
        if cur is None:
            return _ret(None)'''
assert s.count(old) == 1, "def anchor"
s = s.replace(old, new)

# every existing `return None` / `return v` inside the loop has to go through _ret
old = """        r = cur.get("result", -1)
        if r != -1:
            v = 1.0 if r == pilot_i else -1.0"""
new = """        if want_phi and phi is None and cur.get("yourIndex") != pilot_i:
            # first state after our turn handed over: what this line of play built
            phi = _setup_potential(state, pilot_i)
        r = cur.get("result", -1)
        if r != -1:
            v = 1.0 if r == pilot_i else -1.0"""
assert s.count(old) == 1, "turn-boundary anchor"
s = s.replace(old, new)

for a, b in ((
        """                except Exception:                              # noqa: BLE001
                    pass
            return v""",
        """                except Exception:                              # noqa: BLE001
                    pass
            return _ret(v)"""), (
        """        if not ob.get("select"):
            return None""",
        """        if not ob.get("select"):
            return _ret(None)"""), (
        """        try:
            choice = agent(ob)
        except Exception:
            return None""",
        """        try:
            choice = agent(ob)
        except Exception:
            return _ret(None)"""), (
        """        nxt = _raw_step(state["searchId"], choice)
        if nxt.get("error", 0) != 0 or not nxt.get("state"):
            return None""",
        """        nxt = _raw_step(state["searchId"], choice)
        if nxt.get("error", 0) != 0 or not nxt.get("state"):
            return _ret(None)""")):
    assert s.count(a) == 1, "return anchor %r" % a[:40]
    s = s.replace(a, b)

old = """        steps += 1
    return None"""
new = """        steps += 1
    return _ret(None)"""
assert s.count(old) == 1, "tail anchor"
s = s.replace(old, new)

# and the caller: take phi from the playout instead of from the successor
old = """                if want_phi:
                    phis[i].append(_setup_potential(step["state"], pilot_i))
                v = _playout(step["state"], pilot_i, agent_me, agent_opp)
                if v is not None:
                    if shape:
                        # the successor's potential; the root's is common to every candidate in
                        # this scenario and cancels in the pairwise comparison
                        v += SETUP_GAMMA * _setup_potential(step["state"], pilot_i)
                    vals[i].append(v)"""
new = """                if want_phi:
                    v, ph = _playout(step["state"], pilot_i, agent_me, agent_opp, want_phi=True)
                    if ph is not None:
                        phis[i].append(ph)
                else:
                    v = _playout(step["state"], pilot_i, agent_me, agent_opp)
                    ph = None
                if v is not None:
                    if shape and ph is not None:
                        v += SETUP_GAMMA * ph
                    vals[i].append(v)"""
assert s.count(old) == 1, "caller anchor"
s = s.replace(old, new)

open(p + ".new", "w").write(s)
os.replace(p + ".new", p)
print("patched: phi is sampled at our turn boundary inside the playout")

import subprocess
print(subprocess.run(["python3", "-c",
                      "import ast;ast.parse(open('/root/ptcg/repo/tools/rl_branch.py').read());"
                      "print('parses OK')"], capture_output=True, text=True).stdout.strip())
